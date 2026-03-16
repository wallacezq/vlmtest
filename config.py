"""
Central configuration — change model name and paths here only.
"""

# HuggingFace model identifier (or local path after export)
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Default directory for pre-exported OpenVINO IR model
OV_MODEL_DIR = "./ov_qwen3_vl_2b"

# Default OpenVINO device
OV_DEVICE = "GPU"

# Maximum number of concurrent caption inferences on the GPU
MAX_CONCURRENT_CAPTIONS = 3
