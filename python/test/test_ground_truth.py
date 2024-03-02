import torch
import torch.nn as nn

from loftr_pytorch.supervision.ground_truth import warp_grid


def test_warp_grid():
    wc, hc = 4, 4
    B, L, H, W = 2, wc * hc, 32, 32
    pos_x = torch.linspace(0, wc - 1, wc)
    pos_y = torch.linspace(0, hc - 1, hc)
    grid_xy = torch.stack(torch.meshgrid(pos_x, pos_y, indexing="ij")).transpose(1, 2)
    grid_xy = grid_xy.permute(1, 2, 0).reshape(1, hc * wc, -1).repeat(B, 1, 1) * (
        H // hc
    )
    grid_xy = grid_xy.double()
    depth0 = torch.randint(1, 10, (B, H, W)).double()
    T_0to1 = torch.eye(3, 4).unsqueeze(0).repeat(B, 1, 1).double()
    K0 = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).double()
    K1 = K0.clone()

    warped_grid_pts0 = warp_grid(
        grid_xy,
        depth0,
        T_0to1,
        K0,
        K1,
    )

    assert warped_grid_pts0.shape == (B, L, 2)
    assert torch.allclose(warped_grid_pts0, grid_xy, atol=1e-2, rtol=1e-3)
