import torch


def create_meshgrid(h: int, w: int) -> torch.Tensor:
    pos_x = torch.linspace(0, w - 1, w)
    pos_y = torch.linspace(0, h - 1, h)
    grid_xy = torch.stack(torch.meshgrid(pos_x, pos_y, indexing="ij")).transpose(1, 2)
    return grid_xy.permute(1, 2, 0).reshape(1, h * w, -1)


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
    warped_grid_pts0 = warped_grid_pts0_homogeneous[:, :2, :].transpose(1, 2)
    warped_grid_pts0[grid_pts0_depth == 0, :] = float("-inf")

    return warped_grid_pts0


@torch.no_grad()
def spvs_coarse(data, coarse_scale: float):

    scale = 1 / coarse_scale
    scale0 = scale * data["scale0"][:, None] if "scale0" in data else scale
    scale1 = scale * data["scale1"][:, None] if "scale1" in data else scale

    N, _, h0, w0 = data["image0"].shape
    _, _, h1, w1 = data["image1"].shape

    h0_c, w0_c, h1_c, w1_c = map(lambda x: round(x // scale), [h0, w0, h1, w1])

    grid_pt0_c = create_meshgrid(h0_c, w0_c)
    grid_pt0_i = grid_pt0_c * scale0
    grid_pt1_c = create_meshgrid(h1_c, w1_c)
    grid_pt1_i = grid_pt1_c * scale1

    wpts0_i = warp_grid(
        grid_pt0_i.double(), data["depth0"], data["T_0to1"], data["K0"], data["K1"]
    )
    wpts1_i = warp_grid(
        grid_pt1_i.double(), data["depth1"], data["T_1to0"], data["K1"], data["K0"]
    )

    wkpts0_c = wpts0_i / scale0
    wkpts1_c = wpts1_i / scale1

    def out_bound_mask(pt, w, h):
        return (
            (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        )

    wkpts0_c_round = wkpts0_c[:, :, :].round().long()
    wkpts1_c_round = wkpts1_c[:, :, :].round().long()

    if "mask0" in data:
        data["mask0"] = data["mask0"] * (
            ~out_bound_mask(wkpts0_c_round, w1_c, h1_c).view(N, h0_c, w0_c).bool()
        )
    if "mask1" in data:
        data["mask1"] = data["mask1"] * (
            ~out_bound_mask(wkpts1_c_round, w0_c, h0_c).view(N, h1_c, w1_c).bool()
        )

    wkpts0_c_round[data["mask0"].view(N, h0_c * w0_c) == 0, :] = -1000
    wkpts1_c_round[data["mask1"].view(N, h1_c * w1_c) == 0, :] = -1000

    nearest_index1 = (
        wkpts0_c_round[..., 0] + wkpts0_c_round[..., 1] * w1_c
    )  # (N, h0_c * w0_c)
    nearest_index0 = (
        wkpts1_c_round[..., 0] + wkpts1_c_round[..., 1] * w0_c
    )  # (N, h1_c * w1_c)

    loop_back = torch.full_like(nearest_index1, -1)
    valid_mask = nearest_index1 >= 0
    valid_indices = nearest_index1[valid_mask]
    b_indices, i_indices = torch.where(valid_mask)
    loop_back[b_indices, i_indices] = torch.where(
        valid_indices >= 0, nearest_index0[b_indices, valid_indices], -1
    )

    correct_0to1 = loop_back == torch.arange(h0_c * w0_c)[None].repeat(N, 1)
    conf_matrix_gt = torch.zeros(N, h0_c * w0_c, h1_c * w1_c)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1

    # to train coarse-level
    data.update({"conf_matrix_gt": conf_matrix_gt})

    # to train fine-level
    if len(b_ids) == 0:
        print(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        # TODO:: check whether this is a good idea
        b_ids = torch.empty(0)
        i_ids = torch.empty(0)
        j_ids = torch.empty(0)

    data.update({"spv_b_ids": b_ids, "spv_i_ids": i_ids, "spv_j_ids": j_ids})
    data.update({"spv_w_pt0_i": wpts0_i, "spv_pt1_i": grid_pt1_i})
