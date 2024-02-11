import torch

from loftr_pytorch.model.attention import TorchNativeAttention, LinearAttention, FullAttention

B = 4
L = 32 * 32
S = 24 * 24
n_embd = 16
n_heads = 4


def test_FullAttention():
    attn = FullAttention()
    q = torch.randn(B, n_heads, L, n_embd // n_heads)
    k = torch.randn(B, n_heads, S, n_embd // n_heads)
    v = torch.randn(B, n_heads, S, n_embd // n_heads)
    q_mask = torch.rand(B, L)
    kv_mask = torch.rand(B, S)
    y = attn(q, k, v, q_mask, kv_mask)
    assert y.shape == (B, n_heads, L, n_embd // n_heads)


def test_TorchNativeAttention():
    attn = TorchNativeAttention()
    q = torch.randn(B, n_heads, L, n_embd // n_heads)
    k = torch.randn(B, n_heads, S, n_embd // n_heads)
    v = torch.randn(B, n_heads, S, n_embd // n_heads)
    q_mask = torch.rand(B, L)
    kv_mask = torch.rand(B, S)
    y = attn(q, k, v, q_mask, kv_mask)
    assert y.shape == (B, n_heads, L, n_embd // n_heads)


def test_LinearAttention():
    attn = LinearAttention()
    q = torch.randn(B, n_heads, L, n_embd // n_heads)
    k = torch.randn(B, n_heads, S, n_embd // n_heads)
    v = torch.randn(B, n_heads, S, n_embd // n_heads)
    q_mask = torch.rand(B, L)
    kv_mask = torch.rand(B, S)
    y = attn(q, k, v, q_mask, kv_mask)
    assert y.shape == (B, n_heads, L, n_embd // n_heads)
