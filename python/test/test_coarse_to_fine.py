import os
import yaml
import torch
from loftr_pytorch.model.coarse_to_fine import CoarseToFine


def test_coarse_to_fine():
    # set dimensions
    hw0_i = (256, 256)
    hw1_i = (192, 192)
    hw0_c = (32, 32)  # 1/8 of hw0_i
    hw1_c = (24, 24)  # 1/8 of hw1_i
    hw0_f = (128, 128)  # 1/2 of hw0_i
    hw1_f = (96, 96)  # 1/2 of hw1_i
    B = 2
    L = hw0_c[0] * hw0_c[1]
    S = hw1_c[0] * hw1_c[1]

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "loftr_pytorch/config/default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = config["coarse_to_fine"]
    dim_coarse = config["dim_coarse"]
    dim_fine = config["dim_fine"]
    window = config["window"]

    feat_c0, feat_c1 = torch.randn(B, L, dim_coarse), torch.randn(B, S, dim_coarse)
    feat_f1, feat_f0 = torch.randn(B, dim_fine, *hw1_f), torch.randn(
        B, dim_fine, *hw0_f
    )

    batch_ids = torch.tensor([0, 0, 1, 1])
    l_ids = torch.tensor([5, 200, 501, 802])
    s_ids = torch.tensor([105, 201, 320, 405])

    model = CoarseToFine(config)
    feat_f1_unfold, feat_f0_unfold = model(
        feat_f0, feat_f1, feat_c0, feat_c1, batch_ids, l_ids, s_ids
    )
    valid_len = len(batch_ids)
    assert feat_f1_unfold.shape == (valid_len, window**2, dim_fine)
    assert feat_f0_unfold.shape == (valid_len, window**2, dim_fine)
