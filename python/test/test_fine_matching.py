import torch
from loftr_pytorch.matcher.fine_matching import FineMatcher


def test_fine_matcher():
    window = 5
    M = 2
    C = 8

    data = {
        "hw0_i": (512, 512),
        "hw0_f": (256, 256),
        "batch_ids": torch.tensor([0, 1]),
        "l_ids": torch.tensor([5, 200]),
        "s_ids": torch.tensor([105, 405]),
        "mkpts0_c": torch.randint(window - 1, (M, 2)),
        "mkpts1_c": torch.randint(window - 1, (M, 2)),
    }

    feat_f0 = torch.randn(M, window**2, C)
    feat_f1 = torch.randn(M, window**2, C)

    model = FineMatcher(window=window)
    model(feat_f0, feat_f1, data)
    assert data["mkpts0_f"].shape == (M, 2)
    assert data["mkpts1_f"].shape == (M, 2)
