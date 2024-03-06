import os
import yaml
import torch
from loftr_pytorch.model.loftr import LoFTR


def test_compile():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "loftr_pytorch/config/default.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    num_cuda_devices = torch.cuda.device_count()
    device = f"cuda:{num_cuda_devices-1}" if torch.cuda.is_available() else "cpu"

    # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
    # Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
    torch.set_float32_matmul_precision("high")

    with torch.device(device):
        if device.startswith("cuda"):
            device = "cuda"
        model = LoFTR(config).eval().to(device)
        model = torch.compile(model)
        x0 = torch.rand(1, 1, 640, 480).to(device)
        x1 = torch.rand(1, 1, 640, 480).to(device)
        data = {"image0": x0, "image1": x1, 'hw0_i': (640, 480), 'hw1_i': (640, 480)}
        model(data)
