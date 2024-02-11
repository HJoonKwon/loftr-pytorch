import os
import torch

from loftr_pytorch.model.transformer import (
    LoFTREncoderLayer,
    LoFTREncoderLayerPreLN,
    LocalFeatureTransformer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B = 4
L = 32 * 32
S = 24 * 24
n_embd = 16
n_heads = 4
attn_dropout = 0
proj_dropout = 0
ffwd_dropout = 0
layer_names = ["self", "cross"] * 2


def test_LoFTREncoderLayer():
    loftr_encoder_layer = LoFTREncoderLayer(
        n_embd, n_heads, attn_dropout, proj_dropout, ffwd_dropout, "native"
    )
    x = torch.randn(B, L, n_embd)
    source = torch.randn(B, S, n_embd)
    x_mask = torch.rand(B, L)
    source_mask = torch.rand(B, S)
    y = loftr_encoder_layer(x, source, x_mask, source_mask)
    assert y.shape == (B, L, n_embd)

    torch.onnx.export(
        loftr_encoder_layer,
        (x, source, x_mask, source_mask),
        "./model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x", "source", "x_mask", "source_mask"],
        output_names=["result"],
    )

    assert os.path.isfile("./model.onnx")
    os.remove("./model.onnx")


def test_LoFTREncoderLayerPreLN():
    loftr_encoder_layer_post_ln = LoFTREncoderLayerPreLN(
        n_embd, n_heads, attn_dropout, proj_dropout, ffwd_dropout, "native"
    )
    x = torch.randn(B, L, n_embd)
    source = torch.randn(B, S, n_embd)
    x_mask = torch.rand(B, L)
    source_mask = torch.rand(B, S)
    y = loftr_encoder_layer_post_ln(x, source, x_mask, source_mask)
    assert y.shape == (B, L, n_embd)

    torch.onnx.export(
        loftr_encoder_layer_post_ln,
        (x, source, x_mask, source_mask),
        "./model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x", "source", "x_mask", "source_mask"],
        output_names=["result"],
    )

    assert os.path.isfile("./model.onnx")
    os.remove("./model.onnx")


def test_LocalFeatureTransformer():
    local_feature_transformer = LocalFeatureTransformer(
        layer_names, n_embd, n_heads, attn_dropout, proj_dropout, ffwd_dropout
    )
    x = torch.randn(B, L, n_embd)
    source = torch.randn(B, S, n_embd)
    x_mask = torch.rand(B, L)
    source_mask = torch.rand(B, S)
    feat0, feat1 = local_feature_transformer(x, source, x_mask, source_mask)
    assert feat0.shape == (B, L, n_embd)
    assert feat1.shape == (B, S, n_embd)

    torch.onnx.export(
        local_feature_transformer,
        (x, source, x_mask, source_mask),
        "./model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x", "source", "x_mask", "source_mask"],
        output_names=["result"],
    )

    assert os.path.isfile("./model.onnx")
    os.remove("./model.onnx")
