"""
Central configuration — change model name and paths here only.
"""

# ---- Model backend: "qwen", "qwen35", "minicpm", or "internvl" ----
MODEL_BACKEND = "qwen"

# ---- Qwen3-VL settings ----
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OV_MODEL_DIR = "./ov_qwen3_vl_2b"

# ---- Qwen3.5-VL settings ----
QWEN35_MODEL_ID = "Qwen/Qwen3.5-VL-3B-Instruct"
QWEN35_OV_MODEL_DIR = "./ov_qwen35_vl_3b"

# ---- MiniCPM-V settings ----
MINICPM_MODEL_ID = "openbmb/MiniCPM-V-2_6"
MINICPM_OV_MODEL_DIR = "./ov_minicpm_v_2_6"

# ---- InternVL3 settings ----
INTERNVL_MODEL_ID = "OpenGVLab/InternVL3-2B"
INTERNVL_OV_MODEL_DIR = "./ov_internvl3_2b"

# Number of frames to sample from the video buffer for chunk captioning
MINICPM_VIDEO_CHUNK_FRAMES = 8

# Default OpenVINO device
OV_DEVICE = "GPU"

# Maximum number of concurrent caption inferences on the GPU
MAX_CONCURRENT_CAPTIONS = 3
