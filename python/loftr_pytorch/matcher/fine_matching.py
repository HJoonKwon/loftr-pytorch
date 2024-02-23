import torch
import torch.nn as nn
import torch.nn.functional as F


class FineMatcher(nn.Module):
    def __init__(self, window=5):
        super().__init__()
        self.window = window

    def forward(self, feat_f0, feat_f1, mkpts0_c, mkpts1_c):
        """
        Args:
            feat0 (torch.Tensor): [M, window**2, C]
            feat1 (torch.Tensor): [M, window**2, C]
            data (dict)
        Update:
            data (dict):{
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """

        M, WW, C = feat_f0.shape
        window = int(WW**0.5)
        self.window = window
        if M == 0:
            assert (
                self.training == False
            ), "M is always >0, when training, see coarse_matching.py"
            return mkpts0_c, mkpts1_c

        feat_f0_middle = feat_f0[:, window // 2, :]  # (M, C)
        score_matrix = feat_f0_middle[:, None, :] @ feat_f1.transpose(
            -2, -1
        )  # (M, 1, C) @ (M, C, window**2) = (M, 1, window**2)
        score_matrix = score_matrix.squeeze(1) / C**0.5  # (M, window**2)
        score_matrix = F.softmax(score_matrix, dim=1)

        pos_x = torch.linspace(0, window - 1, window, device=feat_f0.device)
        pos_x = (pos_x / (window - 1) - 0.5) * 2.0
        pos_y = pos_x.clone()
        grid_x, grid_y = torch.stack(
            torch.meshgrid(pos_x, pos_y, indexing="ij")
        ).transpose(1, 2)
        grid_x = grid_x.reshape(-1)  # (window**2)
        grid_y = grid_y.reshape(-1)

        expected_x = torch.sum(
            (grid_x * score_matrix), dim=-1, keepdim=True
        )  # vectorization, (M, 1)
        expected_y = torch.sum((grid_y * score_matrix), dim=-1, keepdim=True)  # (M, 1)
        expected_coords = torch.concat([expected_x, expected_y], dim=-1)  # (M,2)

        # TODO:: expec_f for training

        return self._fine_match(expected_coords, mkpts0_c, mkpts1_c)

    @torch.no_grad()
    def _fine_match(self, expected_coords, mkpts0_c, mkpts1_c):

        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + expected_coords * (
            self.window // 2
        )  # TODO:: M and M' are different for training.

        return mkpts0_f, mkpts1_f

    def rescale_mkpts_to_image(self, mkpts0, mkpts1, scale0, scale1):
        return mkpts0 * scale0, mkpts1 * scale1
