"""
Video captioning engine using Qwen3-VL-2B-Instruct with OpenVINO GPU acceleration.
"""

import threading
import time
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from config import MODEL_ID, OV_DEVICE, MAX_CONCURRENT_CAPTIONS

logger = logging.getLogger(__name__)


class VideoCaptioner:
    """Loads Qwen3-VL-2B-Instruct on OpenVINO GPU and captions video frames.

    Uses a semaphore to allow up to MAX_CONCURRENT_CAPTIONS parallel
    inferences on the same GPU-compiled model.
    """

    def __init__(
        self,
        model_path: str = MODEL_ID,
        device: str = OV_DEVICE,
        max_new_tokens: int = 64,
        max_concurrent: int = MAX_CONCURRENT_CAPTIONS,
    ):
        self.max_new_tokens = max_new_tokens
        self._semaphore = threading.Semaphore(max_concurrent)

        logger.info("Loading processor from %s ...", model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        model_dir = Path(model_path)
        export_needed = not (model_dir.is_dir() and (model_dir / "openvino_model.xml").exists())

        logger.info(
            "Loading model on OpenVINO device=%s (export=%s) ...", device, export_needed
        )
        self.model = OVModelForVisualCausalLM.from_pretrained(
            model_path,
            export=export_needed,
            device=device,
            compile=True,
        )
        logger.info("Model ready on %s.", device)

    def caption_frame(self, frame_bgr: np.ndarray) -> str:
        """Generate a caption for a single BGR (OpenCV) frame.

        Thread-safe: up to max_concurrent inferences run in parallel.
        """
        # Convert BGR -> RGB -> PIL
        rgb = frame_bgr[:, :, ::-1]
        pil_image = Image.fromarray(rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {
                        "type": "text",
                        "text": "Describe what is happening in this image in one concise sentence.",
                    },
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        with self._semaphore:
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # Trim the prompt tokens from the output
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        captions = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return captions[0].strip()
