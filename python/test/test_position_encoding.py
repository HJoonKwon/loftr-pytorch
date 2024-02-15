import os
import yaml
import torch
from loftr_pytorch.model.position_embedding import PositionEmbeddingSine


def test_PositionEmbeddingSine():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "loftr_pytorch/config/default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dim = config["transformer"]["position_embedding"]["dim"]
    model = PositionEmbeddingSine(config["transformer"]["position_embedding"])

    x = torch.rand(1, dim, 256, 256)
    feat = model(x)
    assert feat.shape == torch.Size([1, dim, 256, 256])
