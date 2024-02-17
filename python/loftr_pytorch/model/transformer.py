import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FullAttention, TorchNativeAttention, LinearAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_embd, n_heads, attn_dropout=0, proj_dropout=0, attention="native"
    ):
        super().__init__()
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.project = nn.Linear(n_embd, n_embd, bias=False)
        if attention == "linear":
            self.attention = LinearAttention()
        elif attention == "native":
            self.attention = TorchNativeAttention(attn_dropout)
        else:
            self.attention = FullAttention(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.n_heads = n_heads
        self.n_embd = n_embd

    def forward(self, x, source, x_mask=None, source_mask=None):
        B, L, C = x.shape
        q, k, v = x, source, source
        q = q.view(B, -1, self.n_heads, self.n_embd // self.n_heads).transpose(2, 1)
        k = k.view(B, -1, self.n_heads, self.n_embd // self.n_heads).transpose(2, 1)
        v = v.view(B, -1, self.n_heads, self.n_embd // self.n_heads).transpose(2, 1)

        out = self.attention(q, k, v, x_mask, source_mask)
        out = out.transpose(1, 2).reshape(B, L, self.n_embd)
        out = self.project(self.proj_dropout(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd * 2, bias=False),
            nn.ReLU(),
            nn.Linear(2 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FeedForwardPreLN(nn.Module):
    def __init__(self, n_embd, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 2, bias=False),
            nn.ReLU(),
            nn.Linear(2 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LoFTREncoderLayer(nn.Module):
    def __init__(
        self,
        n_embd,
        n_heads,
        attn_dropout,
        proj_dropout,
        ffwd_dropout,
        attention="native",
    ):
        super().__init__()
        self.mhsa = MultiHeadAttention(
            n_embd, n_heads, attn_dropout, proj_dropout, attention
        )
        self.ffwd = FeedForward(n_embd, ffwd_dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, source, x_mask=None, source_mask=None):
        message = self.mhsa(x, source, x_mask, source_mask)
        message = self.ln1(message)

        message = self.ffwd(torch.cat([x, message], dim=2))
        message = self.ln2(message) + x
        return message


class LoFTREncoderLayerPreLN(nn.Module):
    def __init__(
        self,
        n_embd,
        n_heads,
        attn_dropout,
        proj_dropout,
        ffwd_dropout,
        attention="native",
    ):
        super().__init__()
        self.mhsa = MultiHeadAttention(
            n_embd, n_heads, attn_dropout, proj_dropout, attention
        )
        self.ffwd = FeedForwardPreLN(n_embd, ffwd_dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, source, x_mask=None, source_mask=None):
        x = x + self.mhsa(self.ln1(x), self.ln1(source), x_mask, source_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class LocalFeatureTransformer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        n_embd = config["n_embd"]
        self.layer_names = config["layer_names"] * config["n_layers"]
        self.encoders = nn.ModuleList(
            [
                LoFTREncoderLayer(
                    n_embd,
                    config["n_heads"],
                    config["attn_dropout"],
                    config["proj_dropout"],
                    config["ffwd_dropout"],
                    config["attention"],
                )
                for _ in range(len(self.layer_names))
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # only weights

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        for encoder, name in zip(self.encoders, self.layer_names):
            if name == "self":
                feat0 = encoder(feat0, feat0, mask0, mask0)
                feat1 = encoder(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = encoder(feat0, feat1, mask0, mask1)
                feat1 = encoder(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        return feat0, feat1
