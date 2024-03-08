import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/zju3dv/LoFTR/issues/15
def mask_border(input_tensor, border: int, masking_value):
    """Mask borders with value
    Args:
        input_tensor(torch.Tensor): [N, H0, W0, H1, W1]
        border (int)
        masking_value (m.dtype)
    """
    if border <= 0:
        return

    input_tensor[:, :border] = masking_value
    input_tensor[:, :, :border] = masking_value
    input_tensor[:, :, :, :border] = masking_value
    input_tensor[:, :, :, :, :border] = masking_value
    input_tensor[:, -border:] = masking_value
    input_tensor[:, :, -border:] = masking_value
    input_tensor[:, :, :, -border:] = masking_value
    input_tensor[:, :, :, :, -border:] = masking_value


def mask_border_with_padding(
    input_tensor, border, masking_value, padded_mask0, padded_mask1
):
    """Mask borders with value. Here borders are within the unpadded region.
    Args:
        input_tensor(torch.Tensor): [N, H0, W0, H1, W1]
        border (int)
        masking_value (m.dtype)
        padded_mask0(torch.Tensor): [N, H0, W0, H1, W1]
        padded_mask1(torch.Tensor): [N, H0, W0, H1, W1]
    """
    if border <= 0:
        return

    input_tensor[:, :border] = masking_value
    input_tensor[:, :, :border] = masking_value
    input_tensor[:, :, :, :border] = masking_value
    input_tensor[:, :, :, :, :border] = masking_value

    h0s, w0s = (
        padded_mask0.sum(1).max(-1)[0].int(),
        padded_mask0.sum(-1).max(-1)[0].int(),
    )
    h1s, w1s = (
        padded_mask1.sum(1).max(-1)[0].int(),
        padded_mask1.sum(-1).max(-1)[0].int(),
    )
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        input_tensor[b_idx, h0 - border :] = masking_value
        input_tensor[b_idx, :, w0 - border :] = masking_value
        input_tensor[b_idx, :, :, h1 - border :] = masking_value
        input_tensor[b_idx, :, :, :, w1 - border :] = masking_value


def compute_num_max_candidates(mask0, mask1):
    h0s, w0s = mask0.sum(1).max(-1)[0], mask0.sum(-1).max(-1)[0]
    h1s, w1s = mask1.sum(1).max(-1)[0], mask1.sum(-1).max(-1)[0]
    num_max_candidates = torch.stack([h0s * w0s, h1s * w1s], -1).min(-1)[0].sum()
    return num_max_candidates


class CoarseMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.thr = config["thr"]
        self.border_rm = config["border_rm"]
        self.temperature = config["temperature"]
        self.match_type = config["match_type"]
        self.train_num_gt_pad = config["train_num_gt_pad"]
        self.train_coarse_percent = config["train_coarse_percent"]

    def forward(
        self,
        feat0,
        feat1,
        data,
        coarse_gt,
        mask0=None,
        mask1=None,
    ):
        """
        Args:
            feat0 (torch.Tensor): [B, L, C] where L = h0c * w0c
            feat1 (torch.Tensor): [B, S, C] where S = h1c * w1c
            data (dict): meta data and matching results
            mask0 (torch.Tensor): [B, L] (optional). Related to padding.
            mask1 (torch.Tensor): [B, S] (optional). Related to padding.
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'l_ids' (torch.Tensor): [M'], Dimension of L
                's_ids' (torch.Tensor): [M'], Dimension of S
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M = number of predicted matches
            NOTE: M' != M during training.
        """
        B, L, S, C = feat0.shape[0], feat0.shape[1], feat1.shape[1], feat1.shape[-1]
        feat0, feat1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat0, feat1])
        if self.match_type == "dual_softmax":
            sim_matrix = feat0 @ feat1.transpose(-1, -2) / self.temperature  # (B, L, S)
            if mask0 is not None:
                sim_matrix.masked_fill_(
                    mask0[:, :, None] * mask1[:, None, :] == 0, -1e9
                )
            conf_matrix = F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=2)
        else:
            raise NotImplemented

        return self._coarse_match(conf_matrix, data, coarse_gt)

    @torch.no_grad()
    def _coarse_match(self, conf_matrix, data, coarse_gt):
        """
        Args:
            conf_matrix (torch.Tensor): [B, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c'], optionally ['mask0', 'mask1']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'l_ids' (torch.Tensor): [M'],
                's_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        hw0_c, hw1_c = data["hw0_c"], data["hw1_c"]
        B, L, S = conf_matrix.shape
        _device = conf_matrix.device
        assert (
            L == hw0_c[0] * hw0_c[1] and S == hw1_c[0] * hw1_c[1]
        ), "coarse feature dimensions do not match"

        # 1. confidence thresholding
        mask = conf_matrix > self.thr  # (B, L, S)

        # 2. masking border
        mask = mask.view(
            B, hw0_c[0], hw0_c[1], hw1_c[0], hw1_c[1]
        )  # (B, L, S) -> (B, h0c, w0c, h1c, w1c)
        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(
                mask, self.border_rm, False, data["mask0"], data["mask1"]
            )
        mask = mask.view(B, L, S)

        # 3. mutual nearest neighbor
        # (B, L, S) with 1/0 * (B, L, 1) * (B, 1, S) = (B, L, S)
        mask = (
            mask
            * conf_matrix.max(dim=2, keepdim=True)[0]
            * conf_matrix.max(dim=1, keepdim=True)[0]
        )

        # 4. find valid corase matches with their confidence scores
        mask_max_values, all_s_ids = mask.max(dim=2)
        b_ids, l_ids = torch.where(mask_max_values)  # non-zero only
        s_ids = all_s_ids[b_ids, l_ids]  # all_s_ids = (B, L) filled with s_ids
        mconf = conf_matrix[b_ids, l_ids, s_ids]  # (M, ) where M = len(b_ids)

        if self.training:
            if "mask0" in data:
                num_max_candidates = compute_num_max_candidates(
                    data["mask0"], data["mask1"]
                )
            else:
                # NOTE: Original implementation used max instead of min
                num_max_candidates = B * min(L, S)
            num_matches_train = int(num_max_candidates * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_num_gt_pad < num_matches_train

            pred_indices = self._generate_pred_indices(
                num_matches_train, num_matches_pred, _device
            )
            gt_pad_indices = self._generate_gt_pad_indices(
                coarse_gt["spv_b_ids"], num_matches_train, num_matches_pred, _device
            )
            mconf_gt = torch.zeros(len(coarse_gt["spv_b_ids"]), device=_device)

            b_ids, l_ids, s_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, coarse_gt["spv_b_ids"]],
                    [l_ids, coarse_gt["spv_i_ids"]],
                    [s_ids, coarse_gt["spv_j_ids"]],
                    [mconf, mconf_gt],
                ),
            )

        # 5. scale indicies up to original resolution
        scale = data["hw0_i"][0] / data["hw0_c"][0]
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale

        mkpts0 = torch.stack([l_ids % hw0_c[1], l_ids // hw0_c[1]], dim=1) * scale0
        mkpts1 = torch.stack([s_ids % hw1_c[1], s_ids // hw1_c[1]], dim=1) * scale1

        gt_mask = mconf == 0
        m_bids = b_ids[mconf != 0]  # mconf == 0 => gt matches
        mkpts0_c = mkpts0[mconf != 0]
        mkpts1_c = mkpts1[mconf != 0]
        mconf = mconf[mconf != 0]

        prediction = {
            "b_ids": b_ids,
            "l_ids": l_ids,
            "s_ids": s_ids,
            "gt_mask": gt_mask,
            "m_bids": m_bids,
            "mkpts0_c": mkpts0_c,
            "mkpts1_c": mkpts1_c,
            "mconf": mconf,
            "conf_matrix": conf_matrix,
        }

        return prediction

    def _generate_pred_indices(self, num_matches_train, num_matches_pred, device):
        # NOTE: Original implementation used torch.randint which creates duplicate indices
        # which is not what we want
        num_samples = min(num_matches_pred, num_matches_train - self.train_num_gt_pad)
        return torch.randperm(num_matches_pred, device=device)[:num_samples]

    def _generate_gt_pad_indices(
        self, spv_b_ids, num_matches_train, num_matches_pred, device
    ):
        # NOTE: Original implementation used torch.randint which creates duplicate indices
        num_samples = max(self.train_num_gt_pad, num_matches_train - num_matches_pred)
        num_samples = min(len(spv_b_ids), num_samples)
        return torch.randperm(len(spv_b_ids), device=device)[:num_samples]
