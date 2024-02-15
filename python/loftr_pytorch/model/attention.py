import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        B, nh, L, d = q.shape
        B, nh, S, d = k.shape

        scale = 1 / d**0.5
        weight = q @ k.transpose(-2, -1) * scale  # (B, nh, L, S)

        if kv_mask is not None:
            weight = weight.masked_fill(
                q_mask[:, None, :, None] * kv_mask[:, None, None, :] == 0, float("-inf")
            )
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        y = weight @ v  # (B, nh, L, S) @ (B, nh, S, d) = (B, nh, L, d)
        # return y.transpose(1, 2).contiguous()
        return y


class TorchNativeAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.attn_dropout = attn_dropout

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        if kv_mask is not None:
            attn_mask = q_mask[:, None, :, None] * kv_mask[:, None, None, :]
        else:
            attn_mask = None
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout, is_causal=False
        )
        return y


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, q, k, v, q_mask=None, kv_mask=None):

        B, nh, L, d = q.shape
        B, nh, S, d = k.shape
        B, nh, S, dv = v.shape

        q = F.elu(q) + 1
        k = F.elu(k) + 1
        if q_mask is not None:
            q = q * q_mask[:, None, :, None]
        if kv_mask is not None:
            k = k * kv_mask[:, None, :, None]
            v = v * kv_mask[:, None, :, None]

        v = v / S
        kv = k.transpose(-2, -1) @ v  # (B, nh, d, v)
        z = 1 / (
            q @ k.sum(2).unsqueeze(-1) + self.eps
        )  # k.sum(2).unsqueeze(-1) = (B, nh, d, 1) , z = (B, nh, L, 1)
        y = ((q @ kv).unsqueeze(-1) @ z.unsqueeze(-1)).squeeze(-1) * S

        return y
