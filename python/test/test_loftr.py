import os
import yaml
import torch
from loftr_pytorch.model.loftr import LoFTR

num_cuda_devices = torch.cuda.device_count()
device = f"cuda:{num_cuda_devices-1}" if torch.cuda.is_available() else "cpu"

def test_LoFTR():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "loftr_pytorch/config/default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = LoFTR(config).to(device)

    x0 = torch.rand(1, 1, 256, 256, device=device)
    x1 = torch.rand(1, 1, 256, 256, device=device)
    data = {"image0": x0, "image1": x1}
    model(data)
