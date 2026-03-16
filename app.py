"""
Flask application for live video captioning with OpenVINO-accelerated Qwen2-VL.

Usage:
    python app.py                          # webcam (device 0)
    python app.py --source /path/video.mp4 # video file
    python app.py --source rtsp://...      # RTSP / HTTP stream
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

from captioner import VideoCaptioner

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
app = Flask(__name__)

captioner: VideoCaptioner = None          # initialized in main()
cap: cv2.VideoCapture = None              # video capture handle

latest_frame = None                       # most recent BGR frame
frame_lock = threading.Lock()

latest_caption = ""                       # most recent caption text
caption_lock = threading.Lock()

caption_subscribers: list[queue.Queue] = []  # SSE subscriber queues
subscribers_lock = threading.Lock()

captioning_active = False                 # control flag
captioning_thread: threading.Thread = None

CAPTION_INTERVAL = 2.0                    # seconds between caption inferences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------
def _frame_reader():
    """Continuously read frames from the video source."""
    global latest_frame, cap
    while cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video files; for live streams this just retries
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame = frame
        time.sleep(1 / 30)  # ~30 fps pacing


def _captioning_loop():
    """Periodically caption the latest frame and push to subscribers."""
    global latest_caption, captioning_active
    while captioning_active:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.2)
            continue

        try:
            caption = captioner.caption_frame(frame)
        except Exception:
            logger.exception("Captioning failed")
            caption = "[captioning error]"

        with caption_lock:
            latest_caption = caption

        # Push to all SSE subscribers
        event_data = json.dumps({"caption": caption, "ts": time.time()})
        with subscribers_lock:
            for q in caption_subscribers:
                try:
                    q.put_nowait(event_data)
                except queue.Full:
                    pass  # subscriber is slow, skip

        time.sleep(CAPTION_INTERVAL)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


def _generate_mjpeg():
    """Yield MJPEG frames for the /video_feed endpoint."""
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
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


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/caption_stream")
def caption_stream():
    """Server-Sent Events endpoint that pushes new captions."""

    def event_stream():
        q = queue.Queue(maxsize=20)
        with subscribers_lock:
            caption_subscribers.append(q)
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"  # prevent connection timeout
        finally:
            with subscribers_lock:
                caption_subscribers.remove(q)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/caption/start", methods=["POST"])
def start_captioning():
    global captioning_active, captioning_thread
    if captioning_active:
        return jsonify({"status": "already running"})
    captioning_active = True
    captioning_thread = threading.Thread(target=_captioning_loop, daemon=True)
    captioning_thread.start()
    return jsonify({"status": "started"})


@app.route("/caption/stop", methods=["POST"])
def stop_captioning():
    global captioning_active
    captioning_active = False
    return jsonify({"status": "stopped"})


@app.route("/caption/interval", methods=["POST"])
def set_interval():
    global CAPTION_INTERVAL
    data = request.get_json(silent=True) or {}
    try:
        val = float(data.get("interval", CAPTION_INTERVAL))
        if val < 0.5:
            val = 0.5
        CAPTION_INTERVAL = val
    except (TypeError, ValueError):
        return jsonify({"error": "invalid interval"}), 400
    return jsonify({"interval": CAPTION_INTERVAL})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live Video Captioning with OpenVINO")
    p.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0), file path, or stream URL",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model ID or local path to pre-exported OpenVINO model",
    )
    p.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO device (GPU, GPU.0, CPU, ...)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    return p.parse_args()


def main():
    global captioner, cap

    args = parse_args()

    # --- Open video source ----
    source = int(args.source) if args.source.isdigit() else args.source
    logger.info("Opening video source: %s", source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Start frame reader
    threading.Thread(target=_frame_reader, daemon=True).start()

    # --- Load model ---
    captioner = VideoCaptioner(
        model_path=args.model,
        device=args.device,
    )

    # --- Start Flask ---
    logger.info("Starting Flask on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
