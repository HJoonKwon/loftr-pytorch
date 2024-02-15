import os
import yaml
import torch
from loftr_pytorch.model.loftr import LoFTR


def test_LoFTR():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "loftr_pytorch/config/default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = LoFTR(config)

    x0 = torch.rand(1, 1, 256, 256)
    x1 = torch.rand(1, 1, 256, 256)
    data = {"image0": x0, "image1": x1}
    model(data)
