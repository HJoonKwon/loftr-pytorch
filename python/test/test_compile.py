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
    model = LoFTR(config).eval().cuda()
    model = torch.compile(model)
    x0 = torch.rand(1, 1, 640, 480).cuda()
    x1 = torch.rand(1, 1, 640, 480).cuda()
    model(x0, x1)
