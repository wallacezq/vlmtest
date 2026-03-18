"""
Flask application for live video captioning with OpenVINO-accelerated VLMs.

Supports Qwen3-VL (single-frame) and MiniCPM-V (video-chunk) backends.
Up to MAX_CONCURRENT_CAPTIONS simultaneous video streams, each captioned
independently on the same GPU via a shared model.

Usage:
    python app.py                              # Qwen backend (default)
    python app.py --backend minicpm            # MiniCPM video-chunk backend
    python app.py --source 0                   # auto-start with webcam stream
    python app.py --source /path/video.mp4     # auto-start with a video file
    python app.py --model ./ov_model           # pre-exported OpenVINO model dir
    python app.py --device GPU.1               # specific GPU device
"""

import argparse
import collections
import json
import logging
import queue
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

from config import (
    MODEL_ID, QWEN35_MODEL_ID, MINICPM_MODEL_ID, INTERNVL_MODEL_ID, OV_DEVICE,
    MAX_CONCURRENT_CAPTIONS, MODEL_BACKEND,
    MINICPM_VIDEO_CHUNK_FRAMES,
)
from captioner import BaseCaptioner, InferenceStats, create_captioner

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
app = Flask(__name__)

captioner: BaseCaptioner = None           # shared model, initialized in main()
active_backend: str = MODEL_BACKEND       # "qwen" or "minicpm"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream session management
# ---------------------------------------------------------------------------
class StreamSession:
    """Encapsulates one captioned video stream."""

    def __init__(self, stream_id: int, source):
        self.stream_id = stream_id
        self.source_label = str(source)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Frame accumulator for chunk mode — collects all frames between
        # captioning cycles so we can uniformly sample over the full interval.
        self.chunk_size = MINICPM_VIDEO_CHUNK_FRAMES
        self._frame_buffer: list = []
        self._frame_buffer_lock = threading.Lock()

        # Per-stream captioning mode: "frame" (single image) or "chunk" (video)
        self.mode = "frame" if active_backend == "qwen" else "chunk"  # minicpm & internvl default to chunk

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.latest_caption = ""
        self.latest_latency_ms = 0.0
        self.latest_prefill_ms = 0.0
        self.latest_decode_ms = 0.0
        self.latest_decode_tps = 0.0
        self.latest_generated_tokens = 0
        self.inference_count = 0
        self.caption_lock = threading.Lock()

        self.caption_subscribers: list[queue.Queue] = []
        self.subscribers_lock = threading.Lock()

        self.captioning_active = False
        self.caption_interval = 2.0

        # Start background frame reader
        self._reader_thread = threading.Thread(
            target=self._frame_reader, daemon=True
        )
        self._reader_thread.start()

    # --- background workers ---

    def _frame_reader(self):
        while self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue
            with self.frame_lock:
                self.latest_frame = frame
            with self._frame_buffer_lock:
                self._frame_buffer.append(frame.copy())
            time.sleep(1 / 30)

    def _captioning_loop(self):
        while self.captioning_active:
            if self.mode == "chunk":
                # Drain all frames accumulated since last cycle and
                # uniformly sample chunk_size frames across the interval.
                with self._frame_buffer_lock:
                    buffered = self._frame_buffer
                    self._frame_buffer = []
                if len(buffered) > 0 and self.chunk_size > 0:
                    indices = np.linspace(0, len(buffered) - 1, min(self.chunk_size, len(buffered)), dtype=int)
                    frames = [buffered[i] for i in indices]
                else:
                    frames = []
            else:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frames = [self.latest_frame.copy()]
                    else:
                        frames = []

            if not frames:
                time.sleep(0.2)
                continue

            try:
                if self.mode == "chunk":
                    caption, stats = captioner.caption_frames(frames)
                else:
                    caption, stats = captioner.caption_frame(frames[-1])
            except Exception:
                logger.exception("Captioning failed on stream %d", self.stream_id)
                caption = "[captioning error]"
                stats = InferenceStats()

            with self.caption_lock:
                self.latest_caption = caption
                self.latest_latency_ms = stats.latency_ms
                self.latest_prefill_ms = stats.prefill_ms
                self.latest_decode_ms = stats.decode_ms
                self.latest_decode_tps = stats.decode_tps
                self.latest_generated_tokens = stats.generated_tokens
                self.inference_count += 1
                count = self.inference_count

            event_data = json.dumps({
                "stream_id": self.stream_id,
                "caption": caption,
                "ts": time.time(),
                "mode": self.mode,
                "latency_ms": round(stats.latency_ms, 1),
                "prefill_ms": round(stats.prefill_ms, 1),
                "decode_ms": round(stats.decode_ms, 1),
                "decode_tps": round(stats.decode_tps, 1),
                "generated_tokens": stats.generated_tokens,
                "inference_count": count,
            })
            with self.subscribers_lock:
                for q in self.caption_subscribers:
                    try:
                        q.put_nowait(event_data)
                    except queue.Full:
                        pass

            time.sleep(self.caption_interval)

    # --- public controls ---

    def start_captioning(self):
        if self.captioning_active:
            return
        self.captioning_active = True
        threading.Thread(target=self._captioning_loop, daemon=True).start()

    def stop_captioning(self):
        self.captioning_active = False

    def release(self):
        self.captioning_active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# Stream registry  (stream_id -> StreamSession)
streams: dict[int, StreamSession] = {}
streams_lock = threading.Lock()
_next_stream_id = 0


def _allocate_stream_id() -> int:
    global _next_stream_id
    sid = _next_stream_id
    _next_stream_id += 1
    return sid


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", max_streams=MAX_CONCURRENT_CAPTIONS, backend=active_backend)


# ---- Stream CRUD ----

@app.route("/streams", methods=["GET"])
def list_streams():
    with streams_lock:
        info = [
            {
                "stream_id": s.stream_id,
                "source": s.source_label,
                "captioning": s.captioning_active,
                "interval": s.caption_interval,
                "mode": s.mode,
                "chunk_size": s.chunk_size,
                "latency_ms": round(s.latest_latency_ms, 1),
                "prefill_ms": round(s.latest_prefill_ms, 1),
                "decode_ms": round(s.latest_decode_ms, 1),
                "decode_tps": round(s.latest_decode_tps, 1),
                "generated_tokens": s.latest_generated_tokens,
                "inference_count": s.inference_count,
            }
            for s in streams.values()
        ]
    return jsonify(info)


@app.route("/caption/chunk_size/<int:sid>", methods=["POST"])
def set_chunk_size(sid: int):
    with streams_lock:
        session = streams.get(sid)
    if session is None:
        return jsonify({"error": "stream not found"}), 404
    data = request.get_json(silent=True) or {}
    try:
        val = int(data.get("chunk_size", session.chunk_size))
        if val < 1:
            val = 1
        if val > 32:
            val = 32
        session.chunk_size = val
    except (TypeError, ValueError):
        return jsonify({"error": "invalid chunk_size"}), 400
    return jsonify({"chunk_size": session.chunk_size, "stream_id": sid})


@app.route("/streams", methods=["POST"])
def add_stream():
    data = request.get_json(silent=True) or {}
    source_raw = data.get("source", "0")
    source = int(source_raw) if str(source_raw).isdigit() else source_raw

    with streams_lock:
        if len(streams) >= MAX_CONCURRENT_CAPTIONS:
            return jsonify({"error": f"Maximum {MAX_CONCURRENT_CAPTIONS} streams reached"}), 400
        sid = _allocate_stream_id()
        try:
            session = StreamSession(sid, source)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
        streams[sid] = session

    logger.info("Added stream %d from source %s", sid, source)
    return jsonify({"stream_id": sid, "source": session.source_label}), 201


@app.route("/streams/<int:sid>", methods=["DELETE"])
def remove_stream(sid: int):
    with streams_lock:
        session = streams.pop(sid, None)
    if session is None:
        return jsonify({"error": "stream not found"}), 404
    session.release()
    logger.info("Removed stream %d", sid)
    return jsonify({"status": "removed", "stream_id": sid})


# ---- Per-stream video feed ----

@app.route("/video_feed/<int:sid>")
def video_feed(sid: int):
    with streams_lock:
        session = streams.get(sid)
    if session is None:
        return "stream not found", 404

    def generate(sess):
        while True:
            with sess.frame_lock:
                frame = (
                    sess.latest_frame.copy() if sess.latest_frame is not None else None
                )
            if frame is None:
                time.sleep(0.05)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(1 / 30)

    return Response(
        generate(session),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---- Per-stream caption SSE ----

@app.route("/caption_stream/<int:sid>")
def caption_stream(sid: int):
    with streams_lock:
        session = streams.get(sid)
    if session is None:
        return "stream not found", 404

    def event_stream(sess):
        q = queue.Queue(maxsize=20)
        with sess.subscribers_lock:
            sess.caption_subscribers.append(q)
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with sess.subscribers_lock:
                sess.caption_subscribers.remove(q)

    return Response(event_stream(session), mimetype="text/event-stream")


# ---- Per-stream captioning controls ----

@app.route("/caption/start/<int:sid>", methods=["POST"])
def start_captioning(sid: int):
    with streams_lock:
        session = streams.get(sid)
    if session is None:
        return jsonify({"error": "stream not found"}), 404
    session.start_captioning()
    return jsonify({"status": "started", "stream_id": sid})


@app.route("/caption/stop/<int:sid>", methods=["POST"])
def stop_captioning(sid: int):
    with streams_lock:
        session = streams.get(sid)
    if session is None:
        return jsonify({"error": "stream not found"}), 404
    session.stop_captioning()
    return jsonify({"status": "stopped", "stream_id": sid})


@app.route("/caption/interval/<int:sid>", methods=["POST"])
def set_interval(sid: int):
    with streams_lock:
        session = streams.get(sid)
    if session is None:
        return jsonify({"error": "stream not found"}), 404
    data = request.get_json(silent=True) or {}
    try:
        val = float(data.get("interval", session.caption_interval))
        if val < 0.5:
            val = 0.5
        session.caption_interval = val
    except (TypeError, ValueError):
        return jsonify({"error": "invalid interval"}), 400
    return jsonify({"interval": session.caption_interval, "stream_id": sid})


@app.route("/caption/mode/<int:sid>", methods=["POST"])
def set_mode(sid: int):
    with streams_lock:
        session = streams.get(sid)
    if session is None:
        return jsonify({"error": "stream not found"}), 404
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "")
    if mode not in ("frame", "chunk"):
        return jsonify({"error": "mode must be 'frame' or 'chunk'"}), 400
    session.mode = mode
    return jsonify({"mode": session.mode, "stream_id": sid})


@app.route("/info", methods=["GET"])
def server_info():
    return jsonify({"backend": active_backend})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live Video Captioning with OpenVINO")
    p.add_argument(
        "--source",
        default=None,
        help="Video source to auto-add on startup (webcam index, file, or URL)",
    )
    p.add_argument(
        "--backend",
        default=MODEL_BACKEND,
        choices=["qwen", "qwen35", "minicpm", "internvl"],
        help="Captioning backend: qwen (single-frame), qwen35/minicpm/internvl (video-chunk)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID or local OpenVINO model directory "
             "(default: auto-selected per backend)",
    )
    p.add_argument(
        "--device",
        default=OV_DEVICE,
        help="OpenVINO device (GPU, GPU.0, CPU, ...)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    return p.parse_args()


def main():
    global captioner, active_backend

    args = parse_args()
    active_backend = args.backend

    # Pick default model per backend if not explicitly given
    _default_models = {
        "qwen": MODEL_ID,
        "qwen35": QWEN35_MODEL_ID,
        "minicpm": MINICPM_MODEL_ID,
        "internvl": INTERNVL_MODEL_ID,
    }
    if args.model is None:
        model_path = _default_models[args.backend]
    else:
        model_path = args.model

    # --- Load shared model ---
    captioner = create_captioner(
        backend=args.backend,
        model_path=model_path,
        device=args.device,
    )

    # --- Optionally auto-add a stream from CLI ---
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        logger.info("Auto-adding stream from source: %s", source)
        sid = _allocate_stream_id()
        streams[sid] = StreamSession(sid, source)

    # --- Start Flask ---
    logger.info("Starting Flask on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
