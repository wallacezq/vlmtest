#!/usr/bin/env python3
"""
Pre-export VLM models to OpenVINO IR format.

Supports both Qwen3-VL and MiniCPM-V backends.

Usage:
    python export_model.py                                    # export Qwen (default)
    python export_model.py --backend minicpm                  # export MiniCPM-V
    python export_model.py --backend qwen --output ./ov_qwen  # custom output dir
    python export_model.py --backend minicpm --output ./ov_mc  # custom output dir

Then launch the app with:
    python app.py --model ./ov_qwen
    python app.py --backend minicpm --model ./ov_mc
"""

import argparse
import logging

from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor

from config import (
    MODEL_ID, OV_MODEL_DIR,
    MINICPM_MODEL_ID, MINICPM_OV_MODEL_DIR,
    INTERNVL_MODEL_ID, INTERNVL_OV_MODEL_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BACKENDS = {
    "qwen": {"model_id": MODEL_ID, "output": OV_MODEL_DIR, "trust_remote_code": False},
    "minicpm": {"model_id": MINICPM_MODEL_ID, "output": MINICPM_OV_MODEL_DIR, "trust_remote_code": True},
    "internvl": {"model_id": INTERNVL_MODEL_ID, "output": INTERNVL_OV_MODEL_DIR, "trust_remote_code": True},
}


def main():
    parser = argparse.ArgumentParser(description="Export VLM to OpenVINO IR")
    parser.add_argument(
        "--backend",
        default="qwen",
        choices=list(BACKENDS.keys()),
        help="Model backend to export: qwen or minicpm",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (default: auto-selected per backend)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to save the exported model (default: auto-selected per backend)",
    )
    args = parser.parse_args()

    cfg = BACKENDS[args.backend]
    model_id = args.model or cfg["model_id"]
    output_dir = args.output or cfg["output"]
    trust = cfg["trust_remote_code"]

    logger.info("Exporting [%s] %s → %s", args.backend, model_id, output_dir)

    # Export model to OpenVINO IR (this downloads & converts on first run)
    model = OVModelForVisualCausalLM.from_pretrained(
        model_id,
        export=True,
        compile=False,
        trust_remote_code=trust,
    )
    model.save_pretrained(output_dir)

    # Also save processor alongside the model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust)
    processor.save_pretrained(output_dir)

    logger.info("Export complete.  Use:  python app.py --backend %s --model %s", args.backend, output_dir)


if __name__ == "__main__":
    main()
