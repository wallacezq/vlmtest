#!/usr/bin/env python3
"""
Pre-export Qwen3-VL-2B-Instruct to OpenVINO IR format (INT4 weight compression).

Run this once before starting the app if you want faster cold-start times:
    python export_model.py                       # defaults
    python export_model.py --output ./ov_model   # custom output dir

Then launch the app with:
    python app.py --model ./ov_model
"""

import argparse
import logging

from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor

from config import MODEL_ID, OV_MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL to OpenVINO IR")
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        default=OV_MODEL_DIR,
        help="Directory to save the exported model",
    )
    args = parser.parse_args()

    logger.info("Exporting %s → %s", args.model, args.output)

    # Export model to OpenVINO IR (this downloads & converts on first run)
    model = OVModelForVisualCausalLM.from_pretrained(
        args.model,
        export=True,
        compile=False,
    )
    model.save_pretrained(args.output)

    # Also save processor alongside the model
    processor = AutoProcessor.from_pretrained(args.model)
    processor.save_pretrained(args.output)

    logger.info("Export complete.  Use:  python app.py --model %s", args.output)


if __name__ == "__main__":
    main()
