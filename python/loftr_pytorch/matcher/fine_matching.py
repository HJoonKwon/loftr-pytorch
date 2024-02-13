import torch
import torch.nn as nn
import torch.nn.functional as F


class FineMatcher(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def forward(self, feat_f0, feat_f1, data):
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

        M, _, C = feat_f0.shape
        window = self.window
        if M == 0:
            assert (
                self.training == False
            ), "M is always >0, when training, see coarse_matching.py"
            data.update(
                {
                    "mkpts0_f": data["mkpts0_c"],
                    "mkpts1_f": data["mkpts1_c"],
                }
            )
            return

        feat_f0_middle = feat_f0[:, window // 2, :]  # (M, C)
        score_matrix = feat_f0_middle[:, None, :] @ feat_f1.transpose(
            -2, -1
        )  # (M, 1, C) @ (M, C, window**2) = (M, 1, window**2)
        score_matrix = score_matrix.squeeze(1) / C**0.5  # (M, window**2)
        score_matrix = F.softmax(score_matrix, dim=1)

        pos_x = torch.linspace(0, window - 1, window)
        pos_x = (pos_x / (window - 1) - 0.5) * 2.0
        pos_y = pos_x.clone()
        grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing="ij")
        grid_x = grid_x.reshape(-1)  # (window**2)
        grid_y = grid_y.reshape(-1)

        expected_x = torch.sum(
            (grid_x * score_matrix), dim=-1, keepdim=True
        )  # vectorization, (M, 1)
        expected_y = torch.sum((grid_y * score_matrix), dim=-1, keepdim=True)  # (M, 1)
        expected_coords = torch.concat([expected_x, expected_y], dim=-1)  # (M,2)

        # TODO:: expec_f for training

        self._fine_match(expected_coords, data)

    def _fine_match(self, expected_coords, data):
        scale = data["hw0_i"][0] / data["hw0_f"][0]
        scale1 = (
            scale * data["scale1"][data["batch_ids"]] if "scale1" in data else scale
        )

        mkpts0_f = data["mkpts0_c"]
        mkpts1_f = (
            data["mkpts1_c"] + expected_coords * (self.window // 2) * scale1
        )  # TODO:: M and M' are different for training.

        data.update({"mkpts0_f": mkpts0_f, "mkpts1_f": mkpts1_f})
