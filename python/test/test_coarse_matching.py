import torch
from loftr_pytorch.matcher.coarse_matching import CoarseMatcher


def test_coarse_matcher():
    # set dimensions
    hw0_i = (256, 256)
    hw1_i = (192, 192)
    hw0_c = (32, 32)
    hw1_c = (24, 24)
    B = 2
    L = hw0_c[0] * hw0_c[1]
    S = hw1_c[0] * hw1_c[1]
    C = 8

    # generate coarse feature
    feat0, feat1 = torch.randn(B, L, C), torch.randn(B, S, C)
    data = {"hw0_i": hw0_i, "hw1_i": hw1_i, "hw0_c": hw0_c, "hw1_c": hw1_c}

    # generate coarse mask
    mask0 = torch.ones(1, *hw0_c)
    mask0[:, -hw0_c[0] // 3 :, :] = 0
    mask0 = mask0.flatten(-2)
    mask1 = torch.ones(1, *hw1_c)
    mask1[:, :, -hw1_c[1] // 3 :] = 0
    mask1 = mask1.flatten(-2)

    # generate the model
    model = CoarseMatcher(
        thr=0.01, border_rm=2, temperature=0.1, match_type="dual_softmax"
    )

    # perform coarse matching
    model(feat0, feat1, data)
