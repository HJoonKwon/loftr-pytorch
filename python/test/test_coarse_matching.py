import os
import yaml
import torch
from loftr_pytorch.matcher.coarse_matching import CoarseMatcher


def test_coarse_matcher():
    # set dimensions
    hw0_i = (256, 256)
    hw1_i = (192, 192)
    hw0_c = (32, 32)  # 1/8 of hw0_i
    hw1_c = (24, 24)  # 1/8 of hw1_i
    B = 2
    L = hw0_c[0] * hw0_c[1]
    S = hw1_c[0] * hw1_c[1]
    C = 8

    # generate coarse feature
    feat0, feat1 = torch.randn(B, L, C), torch.randn(B, S, C)

    # generate coarse mask
    mask0 = torch.ones(1, *hw0_c)
    mask0[:, -hw0_c[0] // 3 :, :] = 0
    mask0 = mask0.flatten(-2)
    mask1 = torch.ones(1, *hw1_c)
    mask1[:, :, -hw1_c[1] // 3 :] = 0
    mask1 = mask1.flatten(-2)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "loftr_pytorch/config/default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # generate the model
    model = CoarseMatcher(
        config["matcher"]["coarse"],
    )

    scale = hw0_i[0] / hw0_c[0]

    # perform coarse matching
    _, _, _, _, _, mkpts0_c, mkpts1_c, _ = model(feat0, feat1, hw0_c, hw1_c)

    model.rescale_mkpts_to_image(mkpts0_c, mkpts1_c, scale, scale)
