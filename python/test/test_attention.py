import os
import torch

from loftr_pytorch.model.attention import (
    TorchScaledDotProduct,
    LinearAttention,
    FullAttention,
)

B = 4
L = 32 * 32
S = 24 * 24
n_embd = 32
n_heads = 4

num_devices = torch.cuda.device_count()
device = f"cuda:{num_devices-1}" if torch.cuda.is_available() else "cpu"


def test_FullAttention():
    attn = FullAttention()
    q = torch.randn(B, n_heads, L, n_embd // n_heads).to(device)
    k = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    v = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    q_mask = torch.rand(B, L) > 0.1
    kv_mask = torch.rand(B, S) > 0.1
    y = attn(q, k, v, q_mask.to(device), kv_mask.to(device))
    assert y.shape == (B, n_heads, L, n_embd // n_heads)


def test_TorchScaleDotProduct():
    attn = TorchScaledDotProduct()
    q = torch.randn(B, n_heads, L, n_embd // n_heads).to(device)
    k = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    v = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    q_mask = torch.rand(B, L) > 0.1
    kv_mask = torch.rand(B, S) > 0.1

    # test memory efficient attention
    y = attn(q, k, v, q_mask.to(device), kv_mask.to(device))
    assert y.shape == (B, n_heads, L, n_embd // n_heads)

    # test flash attention
    y = attn(q, k, v, None, None)
    assert y.shape == (B, n_heads, L, n_embd // n_heads)


def test_LinearAttention():
    attn = LinearAttention()
    q = torch.randn(B, n_heads, L, n_embd // n_heads).to(device)
    k = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    v = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    q_mask = torch.rand(B, L) > 0.1
    kv_mask = torch.rand(B, S) > 0.1
    y = attn(q, k, v, q_mask.to(device), kv_mask.to(device))
    assert y.shape == (B, n_heads, L, n_embd // n_heads)
