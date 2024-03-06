import torch
from loftr_pytorch.matcher.fine_matching import FineMatcher


def test_fine_matcher():
    window = 5
    M = 2
    C = 8

    hw0_i = (512, 512)
    hw0_f = (256, 256)

    mkpts0_c = torch.randint(window - 1, (M, 2))
    mkpts1_c = torch.randint(window - 1, (M, 2))

    feat_f0 = torch.randn(M, window**2, C)
    feat_f1 = torch.randn(M, window**2, C)

    data = {"hw0_i": hw0_i, "hw0_f": hw0_f}
    coarse_prediction = {"mkpts0_c": mkpts0_c, "mkpts1_c": mkpts1_c}

    model = FineMatcher(window=window)
    fine_prediction = model(feat_f0, feat_f1, coarse_prediction, data)
    mkpts0_f = fine_prediction["mkpts0_f"]
    mkpts1_f = fine_prediction["mkpts1_f"]

    assert mkpts0_f.shape == (M, 2)
    assert mkpts1_f.shape == (M, 2)
