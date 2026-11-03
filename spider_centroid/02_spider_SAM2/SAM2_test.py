# 11/3/26 - Messing with SAM

# ran UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1 uv pip install torch torchvision --index https://download.pytorch.org/whl/cu126 --reinstall

import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)
