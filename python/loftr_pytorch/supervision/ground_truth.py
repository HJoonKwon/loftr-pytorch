import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def warp_grid(grid_pts0, depth0, T_0to1, K0, K1):
    """Warp grid points from I0 to I1 with depth, K and Rt

    Args:
        grid_pts0 (torch.Tensor): [B, L, 2] - <x, y>,
        depth0 (torch.Tensor): [B, H, W],
        T_0to1 (torch.Tensor): [B, 3, 4],
        K0 (torch.Tensor): [B, 3, 3],
        K1 (torch.Tensor): [B, 3, 3],
    Returns:
        warped_grid_pts0 (torch.Tensor): [B, L, 2] <x0_hat, y1_hat>
    """

    B, L, C = grid_pts0.shape
    B, H, W = depth0.shape
    assert C == 2

    ## TODO:: Check whether depth consistency is necessary

    grid_pts0_long = grid_pts0.round().long()
    grid_pts0_depth = torch.stack(
        [depth0[i, grid_pts0_long[i, :, 1], grid_pts0_long[i, :, 0]] for i in range(B)]
    )  # depth for each kpt for each batch
    grid_pts0_homogeneous = torch.concat(
        [grid_pts0, torch.ones_like(grid_pts0[:, :, [0]])], dim=-1
    )
    grid_pts0_unnormalized = grid_pts0_homogeneous * grid_pts0_depth[:, :, None]
    grid_pts0_cam = K0.inverse() @ grid_pts0_unnormalized.transpose(
        -1, -2
    )  # (B, 3, 3) @ (B, 3, L) = (B, 3, L)
    warped_grid_pts0_cam = T_0to1[:, :3, :3] @ grid_pts0_cam + T_0to1[:, :3, [3]]
    warped_grid_pts0_unnormalized = (
        K1 @ warped_grid_pts0_cam
    )  # (B, 3, 3) @ (B, 3, L) = (B, 3, L)
    warped_grid_pts0_homogeneous = warped_grid_pts0_unnormalized / (
        warped_grid_pts0_unnormalized[:, [2], :] + 1e-5
    )
    return warped_grid_pts0_homogeneous[:, :2, :].transpose(1, 2)
