"""
Flask application for live video captioning with OpenVINO-accelerated Qwen3-VL.

Supports up to MAX_CONCURRENT_CAPTIONS simultaneous video streams, each
captioned independently on the same GPU via a shared model.

Usage:
    python app.py                          # start server (add streams via UI)
    python app.py --source 0               # auto-start with webcam stream
    python app.py --source /path/video.mp4 # auto-start with a video file
    python app.py --model ./ov_model       # pre-exported OpenVINO model directory
    python app.py --device GPU.1           # specific GPU device
"""

import argparse
import json
import logging
import queue
import threading
import time

import cv2
from flask import Flask, Response, render_template, request, jsonify

from config import MODEL_ID, OV_DEVICE, MAX_CONCURRENT_CAPTIONS
from captioner import VideoCaptioner

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
app = Flask(__name__)

captioner: VideoCaptioner = None          # shared model, initialized in main()

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

        self.latest_caption = ""
        self.latest_latency_ms = 0.0
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
            time.sleep(1 / 30)

    def _captioning_loop(self):
        while self.captioning_active:
            with self.frame_lock:
                frame = (
                    self.latest_frame.copy() if self.latest_frame is not None else None
                )
            if frame is None:
                time.sleep(0.2)
                continue

            try:
                caption, latency_ms = captioner.caption_frame(frame)
            except Exception:
                logger.exception("Captioning failed on stream %d", self.stream_id)
                caption = "[captioning error]"
                latency_ms = 0.0

            with self.caption_lock:
                self.latest_caption = caption
                self.latest_latency_ms = latency_ms
                self.inference_count += 1
                count = self.inference_count

            event_data = json.dumps({
                "stream_id": self.stream_id,
                "caption": caption,
                "ts": time.time(),
                "latency_ms": round(latency_ms, 1),
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
    return render_template("index.html", max_streams=MAX_CONCURRENT_CAPTIONS)


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
                "latency_ms": round(s.latest_latency_ms, 1),
                "inference_count": s.inference_count,
            }
            for s in streams.values()
        ]
    return jsonify(info)


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
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID or local path to pre-exported OpenVINO model",
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
    global captioner

    args = parse_args()

    # --- Load shared model ---
    captioner = VideoCaptioner(
        model_path=args.model,
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
