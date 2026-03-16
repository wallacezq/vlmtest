# Live Video Captioning with OpenVINO + Qwen3-VL

Real-time video captioning powered by [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) accelerated with [OpenVINO](https://github.com/openvinotoolkit/openvino) on GPU. Supports up to **3 concurrent video streams** captioned simultaneously through a Flask web UI.

## Features

- **OpenVINO GPU acceleration** — model compiled and executed on Intel GPU via OpenVINO runtime
- **Multi-stream support** — caption up to 3 live video feeds concurrently on a single GPU
- **Flexible video sources** — webcam, local video files, RTSP / HTTP streams
- **Real-time captions** — Server-Sent Events push captions to the browser as they're generated
- **Web UI** — dark-themed dashboard with live video feeds, caption overlays, per-stream controls, and a shared caption log
- **Configurable** — model ID, device, concurrency limit, and caption interval all adjustable from a single config file

## Project Structure

```
vlmtest/
├── config.py           # Central configuration (model ID, device, concurrency)
├── captioner.py        # Qwen3-VL model wrapper with OpenVINO GPU inference
├── app.py              # Flask server with multi-stream management
├── export_model.py     # Optional: pre-export model to OpenVINO IR format
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css       # Dark-themed UI styles
└── templates/
    └── index.html      # Frontend with multi-stream grid and caption overlay
```

## Requirements

- Python 3.10+
- Intel GPU with OpenVINO support (or CPU fallback)
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run directly (auto-downloads and converts model on first launch)

```bash
python app.py
```

### 2. Pre-export model for faster startup (optional)

```bash
python export_model.py --output ./ov_qwen3_vl_2b
python app.py --model ./ov_qwen3_vl_2b
```

### 3. Open the UI

Navigate to **http://127.0.0.1:5000** in your browser.

1. Enter a video source in the input bar (e.g. `0` for webcam, a file path, or an RTSP URL)
2. Click **+ Add Stream**
3. Click **▶ Start** on the stream card to begin captioning
4. Repeat for up to 3 streams

## CLI Options

```
python app.py [OPTIONS]

  --source TEXT    Auto-add a video source on startup (webcam index, file, or URL)
  --model TEXT     HuggingFace model ID or local OpenVINO model directory
                   (default: Qwen/Qwen3-VL-2B-Instruct)
  --device TEXT    OpenVINO device — GPU, GPU.0, CPU, etc. (default: GPU)
  --host TEXT      Flask bind address (default: 127.0.0.1)
  --port INT       Flask port (default: 5000)
```

**Examples:**

```bash
# Webcam auto-start
python app.py --source 0

# Video file
python app.py --source /path/to/video.mp4

# RTSP stream on specific GPU
python app.py --source rtsp://192.168.1.10/cam --device GPU.1

# CPU fallback
python app.py --device CPU
```

## Configuration

All key settings live in [`config.py`](config.py):

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-VL-2B-Instruct` | HuggingFace model identifier |
| `OV_MODEL_DIR` | `./ov_qwen3_vl_2b` | Output directory for pre-exported model |
| `OV_DEVICE` | `GPU` | OpenVINO inference device |
| `MAX_CONCURRENT_CAPTIONS` | `3` | Max simultaneous caption inferences on GPU |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/streams` | List active streams |
| `POST` | `/streams` | Add a stream (`{"source": "0"}`) |
| `DELETE` | `/streams/<id>` | Remove a stream |
| `GET` | `/video_feed/<id>` | MJPEG video feed |
| `GET` | `/caption_stream/<id>` | SSE caption stream |
| `POST` | `/caption/start/<id>` | Start captioning |
| `POST` | `/caption/stop/<id>` | Stop captioning |
| `POST` | `/caption/interval/<id>` | Set interval (`{"interval": 2.0}`) |

## Architecture

- **Single shared model** — one `VideoCaptioner` instance loaded on the GPU serves all streams
- **Semaphore-based concurrency** — `threading.Semaphore(MAX_CONCURRENT_CAPTIONS)` gates parallel GPU inference, allowing up to 3 streams to run concurrently without serialization
- **Per-stream isolation** — each stream has its own `StreamSession` with independent video capture, frame buffer, captioning loop, and SSE subscribers
- **Background threads** — frame readers run at ~30 fps; captioning loops run at a configurable interval (default 2s)

## License

This project is provided as-is for educational and research purposes.
