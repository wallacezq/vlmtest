"""
Video captioning engines with OpenVINO GPU acceleration.

Supports:
  - QwenCaptioner  : single-frame captioning via Qwen3-VL
  - MiniCPMCaptioner : video-chunk captioning via MiniCPM-V
"""

import abc
import collections
import threading
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor

from config import (
    MODEL_ID,
    QWEN25_MODEL_ID,
    MINICPM_MODEL_ID,
    INTERNVL_MODEL_ID,
    OV_DEVICE,
    MINICPM_VIDEO_CHUNK_FRAMES,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    """Timing breakdown for a single VLM inference."""
    latency_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    decode_tps: float = 0.0
    generated_tokens: int = 0


class _TokenTimingStreamer:
    """Records a timestamp each time the model produces a token."""

    def __init__(self):
        self.token_timestamps: list[float] = []

    def put(self, value):
        self.token_timestamps.append(time.perf_counter())

    def end(self):
        pass


def _compute_stats(t0: float, t_gen_start: float, streamer: _TokenTimingStreamer) -> InferenceStats:
    """Compute InferenceStats from timing data."""
    latency_ms = (time.perf_counter() - t0) * 1000
    stats = InferenceStats(latency_ms=latency_ms)
    ts = streamer.token_timestamps
    if ts:
        stats.generated_tokens = len(ts)
        stats.prefill_ms = (ts[0] - t_gen_start) * 1000
        if len(ts) > 1:
            stats.decode_ms = (ts[-1] - ts[0]) * 1000
            decode_tokens = len(ts) - 1
            decode_s = ts[-1] - ts[0]
            stats.decode_tps = decode_tokens / decode_s if decode_s > 0 else 0.0
    return stats


def _detect_export_needed(model_path: str) -> bool:
    model_dir = Path(model_path)
    return not (model_dir.is_dir() and any(model_dir.glob("openvino_*.xml")))


def _resolve_model_path(model_path: str) -> str:
    """Resolve local paths to absolute so HuggingFace doesn't mistake them for repo IDs."""
    p = Path(model_path)
    if p.exists():
        return str(p.resolve())
    return model_path


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseCaptioner(abc.ABC):
    """Common interface for all captioning backends."""

    @abc.abstractmethod
    def caption_frame(self, frame_bgr: np.ndarray) -> tuple[str, InferenceStats]:
        """Caption a single BGR frame. Returns (text, stats)."""

    @abc.abstractmethod
    def caption_frames(self, frames_bgr: list[np.ndarray]) -> tuple[str, InferenceStats]:
        """Caption a chunk of BGR frames. Returns (text, stats)."""


# ---------------------------------------------------------------------------
# Qwen3-VL captioner  (single-frame)
# ---------------------------------------------------------------------------
class QwenCaptioner(BaseCaptioner):
    """Qwen3-VL single-frame captioner on OpenVINO."""

    def __init__(
        self,
        model_path: str = MODEL_ID,
        device: str = OV_DEVICE,
        max_new_tokens: int = 64,
    ):
        from qwen_vl_utils import process_vision_info as _pvi
        self._process_vision_info = _pvi

        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        model_path = _resolve_model_path(model_path)

        logger.info("Loading Qwen processor from %s ...", model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        export_needed = _detect_export_needed(model_path)
        logger.info("Loading Qwen model on %s (export=%s) ...", device, export_needed)
        self.model = OVModelForVisualCausalLM.from_pretrained(
            model_path, export=export_needed, device=device, compile=True,
        )
        logger.info("Qwen model ready on %s.", device)

    def caption_frame(self, frame_bgr: np.ndarray) -> tuple[str, InferenceStats]:
        t0 = time.perf_counter()
        pil_image = Image.fromarray(frame_bgr[:, :, ::-1])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Describe what is happening in this image in one concise sentence."},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )

        with self._lock:
            streamer = _TokenTimingStreamer()
            t_gen_start = time.perf_counter()
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, streamer=streamer)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        caption = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return caption, _compute_stats(t0, t_gen_start, streamer)

    def caption_frames(self, frames_bgr: list[np.ndarray]) -> tuple[str, InferenceStats]:
        """Qwen falls back to captioning the last frame only."""
        return self.caption_frame(frames_bgr[-1])


# ---------------------------------------------------------------------------
# Qwen3.5-VL captioner  (video-chunk)
# ---------------------------------------------------------------------------
class Qwen25VLCaptioner(BaseCaptioner):
    """Qwen2.5-VL video-chunk captioner on OpenVINO.

    Uses Qwen's native multi-image/video support via qwen_vl_utils to
    pass multiple frames as a sequence for temporal reasoning.
    """

    def __init__(
        self,
        model_path: str = QWEN25_MODEL_ID,
        device: str = OV_DEVICE,
        max_new_tokens: int = 32,
        chunk_frames: int = MINICPM_VIDEO_CHUNK_FRAMES,
    ):
        from qwen_vl_utils import process_vision_info as _pvi
        self._process_vision_info = _pvi

        self.max_new_tokens = max_new_tokens
        self.chunk_frames = chunk_frames
        self._lock = threading.Lock()

        model_path = _resolve_model_path(model_path)

        logger.info("Loading Qwen2.5-VL processor from %s ...", model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        export_needed = _detect_export_needed(model_path)
        logger.info("Loading Qwen2.5-VL model on %s (export=%s) ...", device, export_needed)
        self.model = OVModelForVisualCausalLM.from_pretrained(
            model_path, export=export_needed, device=device, compile=True,
        )
        logger.info("Qwen2.5-VL model ready on %s.", device)

    def caption_frame(self, frame_bgr: np.ndarray) -> tuple[str, InferenceStats]:
        """Caption a single frame (wraps it as a 1-frame chunk)."""
        return self.caption_frames([frame_bgr])

    def caption_frames(self, frames_bgr: list[np.ndarray]) -> tuple[str, InferenceStats]:
        """Caption a chunk of BGR frames using Qwen's multi-image support."""
        t0 = time.perf_counter()

        indices = np.linspace(0, len(frames_bgr) - 1, min(self.chunk_frames, len(frames_bgr)), dtype=int)
        pil_images = [Image.fromarray(frames_bgr[i][:, :, ::-1]) for i in indices]

        # Build multi-image message for Qwen3.5-VL
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({
            "type": "text",
            "text": "These video frames are in chronological order. "
                    "In 10-15 words, briefly describe the main action.",
        })
        messages = [{"role": "user", "content": content}]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )

        with self._lock:
            streamer = _TokenTimingStreamer()
            t_gen_start = time.perf_counter()
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, streamer=streamer)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        caption = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return caption, _compute_stats(t0, t_gen_start, streamer)


# ---------------------------------------------------------------------------
# MiniCPM-V captioner  (video-chunk)
# ---------------------------------------------------------------------------
class MiniCPMCaptioner(BaseCaptioner):
    """MiniCPM-V video-chunk captioner on OpenVINO.

    Buffers recent frames and sends a chunk of N frames as a multi-image
    prompt so the model can reason about temporal context.
    """

    def __init__(
        self,
        model_path: str = MINICPM_MODEL_ID,
        device: str = OV_DEVICE,
        max_new_tokens: int = 32,
        chunk_frames: int = MINICPM_VIDEO_CHUNK_FRAMES,
    ):
        self.max_new_tokens = max_new_tokens
        self.chunk_frames = chunk_frames
        self._lock = threading.Lock()

        model_path = _resolve_model_path(model_path)

        logger.info("Loading MiniCPM-V processor from %s ...", model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # If the local dir lacks custom processor code, AutoProcessor may
        # return a plain tokenizer.  Fall back to the HuggingFace hub.
        if not hasattr(self.processor, 'image_processor'):
            logger.warning(
                "Local dir lacks MiniCPM-V processor code; "
                "loading processor from %s instead",
                MINICPM_MODEL_ID,
            )
            self.processor = AutoProcessor.from_pretrained(
                MINICPM_MODEL_ID, trust_remote_code=True,
            )
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)

        export_needed = _detect_export_needed(model_path)
        logger.info("Loading MiniCPM-V model on %s (export=%s) ...", device, export_needed)
        self.model = OVModelForVisualCausalLM.from_pretrained(
            model_path, export=export_needed, device=device, compile=True,
            trust_remote_code=True,
        )
        logger.info("MiniCPM-V model ready on %s.", device)

    def _build_prompt(self, pil_images: list[Image.Image]) -> str:
        """Build a text prompt with image placeholders for MiniCPM-V-2.6."""
        image_tags = "".join("(<image>./</image>)" for _ in pil_images)
        question = (
            "These video frames are in chronological order. "
            "In 10-15 words, briefly describe the main action."
        )
        return f"<用户>{image_tags}{question}<AI>"

    def caption_frame(self, frame_bgr: np.ndarray) -> tuple[str, InferenceStats]:
        """Caption a single frame (wraps it as a 1-frame chunk)."""
        return self.caption_frames([frame_bgr])

    def caption_frames(self, frames_bgr: list[np.ndarray]) -> tuple[str, InferenceStats]:
        """Caption a chunk of BGR frames."""
        t0 = time.perf_counter()

        # Sample down to chunk_frames if we have more
        indices = np.linspace(0, len(frames_bgr) - 1, min(self.chunk_frames, len(frames_bgr)), dtype=int)
        pil_images = [Image.fromarray(frames_bgr[i][:, :, ::-1]) for i in indices]

        text_prompt = self._build_prompt(pil_images)
        try:
            inputs = self.processor(
                text=[text_prompt], images=[pil_images],
                padding=True, return_tensors="pt",
            )
        except TypeError:
            # Older MiniCPM-V processor uses different argument names
            inputs = self.processor(
                prompts=[text_prompt], img_list=[pil_images],
                return_tensors="pt",
            )

        with self._lock:
            streamer = _TokenTimingStreamer()
            t_gen_start = time.perf_counter()
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, streamer=streamer)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        caption = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return caption, _compute_stats(t0, t_gen_start, streamer)


# ---------------------------------------------------------------------------
# InternVL3 captioner  (video-chunk)
# ---------------------------------------------------------------------------
class InternVLCaptioner(BaseCaptioner):
    """InternVL3 video-chunk captioner on OpenVINO.

    Sends a chunk of frames as a multi-image prompt using InternVL3's
    <image> placeholder format.
    """

    def __init__(
        self,
        model_path: str = INTERNVL_MODEL_ID,
        device: str = OV_DEVICE,
        max_new_tokens: int = 32,
        chunk_frames: int = MINICPM_VIDEO_CHUNK_FRAMES,
    ):
        self.max_new_tokens = max_new_tokens
        self.chunk_frames = chunk_frames
        self._lock = threading.Lock()

        model_path = _resolve_model_path(model_path)
        hub_id = INTERNVL_MODEL_ID  # always available for processor code

        # InternVL3 doesn't ship a combined processor — load tokenizer and
        # image processor separately.
        logger.info("Loading InternVL3 tokenizer from %s ...", model_path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True)

        logger.info("Loading InternVL3 image processor ...")
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self.image_processor = AutoImageProcessor.from_pretrained(hub_id, trust_remote_code=True)

        export_needed = _detect_export_needed(model_path)
        logger.info("Loading InternVL3 model on %s (export=%s) ...", device, export_needed)
        self.model = OVModelForVisualCausalLM.from_pretrained(
            model_path, export=export_needed, device=device, compile=True,
            trust_remote_code=True,
        )
        # InternVL3 config: how many visual tokens per image tile
        self.num_image_token = getattr(self.model.config, 'num_image_token', 256)
        self.img_context_token = '<IMG_CONTEXT>'
        logger.info("InternVL3 model ready on %s (num_image_token=%d).", device, self.num_image_token)

    def _build_prompt(self, pil_images: list[Image.Image], tiles_per_image: list[int]) -> str:
        """Build a text prompt with expanded IMG_CONTEXT placeholders."""
        parts = []
        for i, n_tiles in enumerate(tiles_per_image):
            n_tokens = n_tiles * self.num_image_token
            parts.append(f"Image-{i+1}: " + self.img_context_token * n_tokens + "\n")
        question = (
            "These video frames are in chronological order. "
            "In 10-15 words, briefly describe the main action."
        )
        user_content = "".join(parts) + question
        messages = [{"role": "user", "content": user_content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def caption_frame(self, frame_bgr: np.ndarray) -> tuple[str, InferenceStats]:
        """Caption a single frame (wraps it as a 1-frame chunk)."""
        return self.caption_frames([frame_bgr])

    def caption_frames(self, frames_bgr: list[np.ndarray]) -> tuple[str, InferenceStats]:
        """Caption a chunk of BGR frames."""
        import torch
        t0 = time.perf_counter()

        indices = np.linspace(0, len(frames_bgr) - 1, min(self.chunk_frames, len(frames_bgr)), dtype=int)
        pil_images = [Image.fromarray(frames_bgr[i][:, :, ::-1]) for i in indices]

        # Process each image individually to know per-image tile count
        all_pixel_values = []
        tiles_per_image = []
        for img in pil_images:
            img_out = self.image_processor(images=img, return_tensors="pt")
            pv = img_out["pixel_values"]  # (num_tiles, C, H, W)
            all_pixel_values.append(pv)
            tiles_per_image.append(pv.shape[0])

        # Build prompt with correct number of IMG_CONTEXT tokens per image
        text_prompt = self._build_prompt(pil_images, tiles_per_image)
        text_inputs = self.tokenizer(text_prompt, return_tensors="pt")

        pixel_values = torch.cat(all_pixel_values, dim=0)
        inputs = {**text_inputs, "pixel_values": pixel_values}

        with self._lock:
            streamer = _TokenTimingStreamer()
            t_gen_start = time.perf_counter()
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, streamer=streamer)

        trimmed = [out[len(inp):] for inp, out in zip(text_inputs.input_ids, generated_ids)]
        caption = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return caption, _compute_stats(t0, t_gen_start, streamer)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_captioner(backend: str = "qwen", **kwargs) -> BaseCaptioner:
    """Create a captioner instance by backend name."""
    if backend == "qwen":
        return QwenCaptioner(**kwargs)
    elif backend == "qwen25":
        return Qwen25VLCaptioner(**kwargs)
    elif backend == "minicpm":
        return MiniCPMCaptioner(**kwargs)
    elif backend == "internvl":
        return InternVLCaptioner(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'qwen', 'qwen25', 'minicpm', or 'internvl'.")
