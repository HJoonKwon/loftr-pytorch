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


def test_FullAttention():
    attn = FullAttention()
    q = torch.randn(B, n_heads, L, n_embd // n_heads)
    k = torch.randn(B, n_heads, S, n_embd // n_heads)
    v = torch.randn(B, n_heads, S, n_embd // n_heads)
    q_mask = torch.rand(B, L) > 0.1
    kv_mask = torch.rand(B, S) > 0.1
    y = attn(q, k, v, q_mask, kv_mask)
    assert y.shape == (B, n_heads, L, n_embd // n_heads)

    torch.onnx.export(
        attn,
        (q, k, v, q_mask, kv_mask),
        "./model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["q", "k", "v", "q_mask", "kv_mask"],
        output_names=["result"],
    )

    assert os.path.isfile("./model.onnx")
    os.remove("./model.onnx")


def test_TorchScaleDotProduct():
    attn = TorchScaledDotProduct()
    q = torch.randn(B, n_heads, L, n_embd // n_heads)
    k = torch.randn(B, n_heads, S, n_embd // n_heads)
    v = torch.randn(B, n_heads, S, n_embd // n_heads)
    q_mask = torch.rand(B, L) > 0.1
    kv_mask = torch.rand(B, S) > 0.1
    y = attn(q, k, v, q_mask, kv_mask)
    assert y.shape == (B, n_heads, L, n_embd // n_heads)

    # torch.onnx.export(
    #     attn,
    #     (q, k, v, q_mask, kv_mask),
    #     "./model.onnx",
    #     export_params=True,
    #     opset_version=17,
    #     do_constant_folding=True,
    #     input_names=["q", "k", "v", "q_mask", "kv_mask"],
    #     output_names=["result"],
    # )

    # assert os.path.isfile("./model.onnx")
    # os.remove("./model.onnx")


def test_LinearAttention():
    attn = LinearAttention()
    q = torch.randn(B, n_heads, L, n_embd // n_heads)
    k = torch.randn(B, n_heads, S, n_embd // n_heads)
    v = torch.randn(B, n_heads, S, n_embd // n_heads)
    q_mask = torch.rand(B, L) > 0.1
    kv_mask = torch.rand(B, S) > 0.1
    y = attn(q, k, v, q_mask, kv_mask)
    assert y.shape == (B, n_heads, L, n_embd // n_heads)

    torch.onnx.export(
        attn,
        (q, k, v, q_mask, kv_mask),
        "./model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["q", "k", "v", "q_mask", "kv_mask"],
        output_names=["result"],
    )

    assert os.path.isfile("./model.onnx")
    os.remove("./model.onnx")
