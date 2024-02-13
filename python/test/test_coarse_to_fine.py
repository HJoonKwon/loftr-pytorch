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
    dim_coarse = 8
    dim_fine = 4
    window = 5

    feat_c0, feat_c1 = torch.randn(B, L, dim_coarse), torch.randn(B, S, dim_coarse)
    feat_f1, feat_f0 = torch.randn(B, dim_fine, *hw1_f), torch.randn(
        B, dim_fine, *hw0_f
    )

    data = {
        "batch_ids": torch.tensor([0, 0, 1, 1]),
        "l_ids": torch.tensor([5, 200, 501, 802]),
        "s_ids": torch.tensor([105, 201, 320, 405]),
    }
    model = CoarseToFine(window=window, dim_coarse=dim_coarse, dim_fine=dim_fine)
    feat_f1_unfold, feat_f0_unfold = model(feat_f0, feat_f1, feat_c0, feat_c1, data)
    valid_len = len(data["batch_ids"])
    assert feat_f1_unfold.shape == (valid_len, window**2, dim_fine)
    assert feat_f0_unfold.shape == (valid_len, window**2, dim_fine)
