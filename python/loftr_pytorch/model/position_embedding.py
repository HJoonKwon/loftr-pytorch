import torch
import torch.nn as nn
import math


# refer to https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/models/position_encoding.py
# TODO:: Need investigation. Why not using dynamic shape?
class PositionEmbeddingSine(nn.Module):
    def __init__(
        self, d_model, max_shape, temperature=10000.0, normalize=True, scale=None
    ):
        super().__init__()
        if scale is None:
            scale = 2 * math.pi

        dim_t = torch.arange(d_model // 2, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / d_model)

        x_embed = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        y_embed = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)

        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.register_buffer("pe", pos, persistent=False)

    def forward(self, x):
        _, _, H, W = x.shape
        return x + self.pe[:, :, :H, :W]
