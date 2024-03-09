import torch
import torch.nn as nn
import torch.nn.functional as F


class FineMatcher(nn.Module):
    def __init__(self, window=5):
        super().__init__()
        self.window = window

    def forward(self, feat_f0, feat_f1, coarse_prediction, data):
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

        mkpts0_c, mkpts1_c = (
            coarse_prediction["mkpts0_c"],
            coarse_prediction["mkpts1_c"],
        )
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

        grid_xy = (
            torch.stack(torch.meshgrid(pos_x, pos_y, indexing="ij"))
            .transpose(1, 2)
            .reshape(-1, window**2)
        )  # (2, window**2)
        expected_xy = (
            score_matrix @ grid_xy.T
        )  # (M, window**2) @ (window**2, 2) = (M, 2)

        var_xy = score_matrix @ (grid_xy**2).T - (expected_xy**2)  # (M, 2)
        std_xy = torch.clamp(var_xy.sum(-1), min=1e-10).sqrt()
        expect_f = torch.cat([expected_xy, std_xy[:, None]], dim=-1)  # (M, 3)

        scale = data["hw0_i"][0] / data["hw0_f"][0]
        scale1 = (
            scale * data["scale1"][coarse_prediction["b_ids"]]
            if "scale1" in data
            else scale
        )

        mkpts0_f, mkpts1_f = self._fine_match(expected_xy, mkpts0_c, mkpts1_c, scale1)

        fine_prediction = {}
        fine_prediction["mkpts0_f"] = mkpts0_f
        fine_prediction["mkpts1_f"] = mkpts1_f
        fine_prediction["expec_f"] = expect_f
        return fine_prediction

    @torch.no_grad()
    def _fine_match(self, expected_coords, mkpts0_c, mkpts1_c, scale1):
        mkpts0_f = mkpts0_c
        mkpts1_f = (
            mkpts1_c + (expected_coords * (self.window // 2) * scale1)[: len(mkpts1_c)]
        )
        return mkpts0_f, mkpts1_f
