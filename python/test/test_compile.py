import os
import yaml
import torch
from loftr_pytorch.model.loftr import LoFTR

num_cuda_devices = torch.cuda.device_count()
device = f"cuda:{num_cuda_devices-1}" if torch.cuda.is_available() else "cpu"


def test_compile():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "loftr_pytorch/config/default.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with torch.device(device):
        model = LoFTR(config).eval().cuda()
        model = torch.compile(model)
        x0 = torch.rand(1, 1, 640, 480).cuda()
        x1 = torch.rand(1, 1, 640, 480).cuda()
        model(x0, x1)
