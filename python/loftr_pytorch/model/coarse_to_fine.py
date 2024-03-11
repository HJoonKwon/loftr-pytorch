import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseToFine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window = config["window"]
        self.is_concat_enabled = config["is_concat_enabled"]
        self.stride = config["coarse_to_fine_ratio"]
        self.dim_coarse = config["dim_coarse"]
        self.dim_fine = config["dim_fine"]

        if self.is_concat_enabled:
            self.coarse_to_fine_proj = nn.Linear(
                self.dim_coarse, self.dim_fine, bias=True
            )
            self.merge_features = nn.Linear(2 * self.dim_fine, self.dim_fine, bias=True)
            self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, coarse_prediction):
        """
        Args:
            feat_f0 (torch.Tensor): [B, dim_fine, *hw0f]
            feat_f1 (torch.Tensor): [B, dim_fine, *hw1f]
            feat_c0 (torch.Tensor): [B, L, dim_coarse]
            feat_c1 (torch.Tensor): [B, S, dim_coarse]
            data (dict): meta data and matching results
        Returns:
            feat_f0_unfold (torch.Tensor): [M, window**2, dim_fine]
            feat_f1_unfold (torch.Tensor): [M, window**2, dim_fine]
        """

        B, L, S = feat_c0.shape[0], feat_c0.shape[1], feat_c1.shape[1]

        b_ids, l_ids, s_ids = (
            coarse_prediction["b_ids"],
            coarse_prediction["l_ids"],
            coarse_prediction["s_ids"],
        )

        if b_ids.shape[0] == 0:
            feat0 = torch.empty(0, self.window**2, self.dim_fine, device=feat_f0.device)
            feat1 = torch.empty(0, self.window**2, self.dim_fine, device=feat_f0.device)
            return feat0, feat1

        feat_f0_unfold = (
            F.unfold(
                feat_f0,
                kernel_size=(self.window, self.window),
                stride=self.stride,
                padding=self.window // 2,
            )
            .transpose(-1, -2)
            .view(B, L, self.window**2, self.dim_fine)
        )
        feat_f0_unfold = feat_f0_unfold[b_ids, l_ids]  # (M, window**2, dim_fine)

        feat_f1_unfold = (
            F.unfold(
                feat_f1,
                kernel_size=(self.window, self.window),
                stride=self.stride,
                padding=self.window // 2,
            )
            .transpose(-1, -2)
            .view(B, S, self.window**2, self.dim_fine)
        )
        feat_f1_unfold = feat_f1_unfold[b_ids, s_ids]  # (M, window**2, dim_fine)

        if self.is_concat_enabled:
            feat_c0_valid = feat_c0[b_ids, l_ids]  # (M, dim_coarse)
            feat_c1_valid = feat_c1[b_ids, s_ids]

            # (2*M, dim_coarse) -> (2*M, dim_fine)
            feat_c_projected = self.coarse_to_fine_proj(
                torch.cat([feat_c0_valid, feat_c1_valid], dim=0)
            )

            # (2*M, dim_fine) -> (2*M, window**2, dim_fine)
            feat_c_projected = feat_c_projected[:, None, :].repeat(1, self.window**2, 1)

            # (2*M, window**2, dim_fine)
            feat_f = torch.cat([feat_f0_unfold, feat_f1_unfold], dim=0)

            # (2*M, window**2, dim_fine) -> (2*M, window**2, dim_fine)
            feat_merged = self.merge_features(
                torch.concat([feat_c_projected, feat_f], dim=-1)
            )

            # (M, window**2, dim_fine), (M, window**2, dim_fine)
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_merged, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold
