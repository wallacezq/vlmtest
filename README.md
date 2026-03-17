# Live Video Captioning with OpenVINO

Real-time video captioning accelerated with [OpenVINO](https://github.com/openvinotoolkit/openvino) on GPU. Supports three VLM backends:

- **[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)** — single-frame captioning
- **[MiniCPM-V-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6)** — video-chunk captioning (multi-frame temporal reasoning)
- **[InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B)** — video-chunk captioning (multi-frame temporal reasoning)

Supports up to **3 concurrent video streams** captioned simultaneously through a Flask web UI.

## Features

- **Triple VLM backends** — Qwen3-VL for per-frame captions, MiniCPM-V and InternVL3 for video-chunk captions with temporal context
- **OpenVINO GPU acceleration** — models compiled and executed on Intel GPU via OpenVINO runtime
- **Multi-stream support** — caption up to 3 live video feeds concurrently on a single GPU
- **Per-stream mode toggle** — switch between Frame and Chunk captioning mode per stream at runtime
- **Flexible video sources** — webcam, local video files, RTSP / HTTP streams
- **Real-time captions** — Server-Sent Events push captions to the browser as they're generated
- **Inference stats** — live display of latency, prefill time, decode time, tokens/sec, and inference count
- **Web UI** — dark-themed dashboard with live video feeds, caption overlays, per-stream controls, and a shared caption log
- **Configurable** — model IDs, device, concurrency limit, chunk size, and caption interval all adjustable from a single config file

## Project Structure

```
vlmtest/
├── config.py           # Central configuration (model IDs, device, concurrency)
├── captioner.py        # VLM backends: QwenCaptioner, MiniCPMCaptioner, InternVLCaptioner
├── app.py              # Flask server with multi-stream management
├── export_model.py     # Pre-export models to OpenVINO IR format
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

### 1. Run with Qwen3-VL (default, single-frame captioning)

```bash
python app.py
```

### 2. Run with MiniCPM-V (video-chunk captioning)

```bash
python app.py --backend minicpm
```

### 3. Run with InternVL3 (video-chunk captioning)

```bash
python app.py --backend internvl
```

### 4. Pre-export models for faster startup (optional)

```bash
# Export Qwen3-VL
python export_model.py --backend qwen

# Export MiniCPM-V
python export_model.py --backend minicpm

# Export InternVL3
python export_model.py --backend internvl

# Run with pre-exported model
python app.py --model ./ov_qwen3_vl_2b
python app.py --backend minicpm --model ./ov_minicpm_v_2_6
python app.py --backend internvl --model ./ov_internvl3_2b
```

### 5. Open the UI

Navigate to **http://127.0.0.1:5000** in your browser.

1. Enter a video source in the input bar (e.g. `0` for webcam, a file path, or an RTSP URL)
2. Click **+ Add Stream**
3. Select **Frame** or **Chunk** mode from the dropdown
4. Click **▶ Start** on the stream card to begin captioning
5. Repeat for up to 3 streams

## CLI Options

### `app.py`

```
python app.py [OPTIONS]

  --backend TEXT   Captioning backend: qwen (single-frame), minicpm or internvl
                   (video-chunk)  (default: qwen)
  --source TEXT    Auto-add a video source on startup (webcam index, file, or URL)
  --model TEXT     HuggingFace model ID or local OpenVINO model directory
                   (default: auto-selected per backend)
  --device TEXT    OpenVINO device — GPU, GPU.0, CPU, etc. (default: GPU)
  --host TEXT      Flask bind address (default: 127.0.0.1)
  --port INT       Flask port (default: 5000)
```

### `export_model.py`

```
python export_model.py [OPTIONS]

  --backend TEXT   Model backend to export: qwen, minicpm, or internvl (default: qwen)
  --model TEXT     HuggingFace model ID (default: auto-selected per backend)
  --output TEXT    Output directory (default: auto-selected per backend)
```

**Examples:**

```bash
# Qwen with webcam auto-start
python app.py --source 0

# MiniCPM-V with video file
python app.py --backend minicpm --source /path/to/video.mp4

# InternVL3 with webcam
python app.py --backend internvl --source 0

# RTSP stream on specific GPU
python app.py --source rtsp://192.168.1.10/cam --device GPU.1

# CPU fallback
python app.py --device CPU
```

## Configuration

All key settings live in [`config.py`](config.py):

| Variable | Default | Description |
|---|---|---|
| `MODEL_BACKEND` | `qwen` | Active backend: `qwen`, `minicpm`, or `internvl` |
| `MODEL_ID` | `Qwen/Qwen3-VL-2B-Instruct` | Qwen model identifier |
| `OV_MODEL_DIR` | `./ov_qwen3_vl_2b` | Qwen export directory |
| `MINICPM_MODEL_ID` | `openbmb/MiniCPM-V-2_6` | MiniCPM-V model identifier |
| `MINICPM_OV_MODEL_DIR` | `./ov_minicpm_v_2_6` | MiniCPM-V export directory |
| `INTERNVL_MODEL_ID` | `OpenGVLab/InternVL3-2B` | InternVL3 model identifier |
| `INTERNVL_OV_MODEL_DIR` | `./ov_internvl3_2b` | InternVL3 export directory |
| `MINICPM_VIDEO_CHUNK_FRAMES` | `8` | Frames sampled per video chunk |
| `OV_DEVICE` | `GPU` | OpenVINO inference device |
| `MAX_CONCURRENT_CAPTIONS` | `3` | Max simultaneous streams on GPU |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/info` | Server info (active backend) |
| `GET` | `/streams` | List active streams |
| `POST` | `/streams` | Add a stream (`{"source": "0"}`) |
| `DELETE` | `/streams/<id>` | Remove a stream |
| `GET` | `/video_feed/<id>` | MJPEG video feed |
| `GET` | `/caption_stream/<id>` | SSE caption stream |
| `POST` | `/caption/start/<id>` | Start captioning |
| `POST` | `/caption/stop/<id>` | Stop captioning |
| `POST` | `/caption/interval/<id>` | Set interval (`{"interval": 2.0}`) |
| `POST` | `/caption/mode/<id>` | Set mode (`{"mode": "frame"}` or `{"mode": "chunk"}`) |
| `POST` | `/caption/chunk_size/<id>` | Set chunk size (`{"chunk_size": 8}`) |

## Architecture

- **Triple backends** — `QwenCaptioner` (single-frame), `MiniCPMCaptioner` (video-chunk), and `InternVLCaptioner` (video-chunk) share a common `BaseCaptioner` interface, selected via `create_captioner()` factory
- **Single shared model** — one captioner instance loaded on the GPU serves all streams
- **Serialized GPU inference** — `threading.Lock` ensures only one inference runs at a time, preventing "Infer Request is busy" errors while all streams remain concurrent in frame capture and SSE delivery
- **Per-stream isolation** — each `StreamSession` has independent video capture, frame ring buffer, captioning loop, mode toggle, and SSE subscribers
- **Video-chunk mode** — the frame reader fills a buffer; in chunk mode, evenly-sampled frames spanning the full captioning interval are sent as a multi-image prompt for temporal reasoning
- **Inference stats** — a `_TokenTimingStreamer` records per-token timestamps to compute prefill time, decode time, and tokens/sec
- **Background threads** — frame readers run at ~30 fps; captioning loops run at a configurable interval (default 2s)

## License

This project is provided as-is for educational and research purposes.
