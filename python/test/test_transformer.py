import os
import yaml
import torch

from loftr_pytorch.model.transformer import (
    LoFTREncoderLayer,
    LoFTREncoderLayerPreLN,
    LocalFeatureTransformer,
)

num_cuda_devices = torch.cuda.device_count()
device = torch.device(f"cuda:{num_cuda_devices-1}" if torch.cuda.is_available() else "cpu")


def test_LoFTREncoderLayer():
    B = 4
    L = 32 * 32
    S = 24 * 24
    n_embd = 32
    n_heads = 4
    attn_dropout = 0
    proj_dropout = 0
    ffwd_dropout = 0
    loftr_encoder_layer = LoFTREncoderLayer(
        n_embd, n_heads, attn_dropout, proj_dropout, ffwd_dropout, "native"
    ).to(device)
    x = torch.randn(B, L, n_embd).to(device)
    source = torch.randn(B, S, n_embd).to(device)
    x_mask = torch.rand(B, L) > 0.1
    source_mask = torch.rand(B, S) > 0.1
    y = loftr_encoder_layer(x, source, x_mask.to(device), source_mask.to(device))
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
    B = 4
    L = 32 * 32
    S = 24 * 24
    n_embd = 32 
    n_heads = 4
    attn_dropout = 0
    proj_dropout = 0
    ffwd_dropout = 0
    loftr_encoder_layer_post_ln = LoFTREncoderLayerPreLN(
        n_embd, n_heads, attn_dropout, proj_dropout, ffwd_dropout, "native"
    ).to(device)
    x = torch.randn(B, L, n_embd).to(device)
    source = torch.randn(B, S, n_embd).to(device)
    x_mask = torch.rand(B, L) > 0.1
    source_mask = torch.rand(B, S) > 0.1
    y = loftr_encoder_layer_post_ln(x, source, x_mask.to(device), source_mask.to(device))
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


def test_LocalFeatureTransformer_coarse():
    B = 4
    L = 32 * 32
    S = 24 * 24
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "loftr_pytorch/config/default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    local_feature_transformer = LocalFeatureTransformer(config["transformer"]["coarse"]).to(device)

    n_embd = config["transformer"]["coarse"]["n_embd"]
    x = torch.randn(B, L, n_embd).to(device)
    source = torch.randn(B, S, n_embd).to(device)
    x_mask = torch.rand(B, L) > 0.1
    source_mask = torch.rand(B, S) > 0.1
    feat0, feat1 = local_feature_transformer(x, source, x_mask.to(device), source_mask.to(device))
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


def test_LocalFeatureTransformer_fine():
    B = 4
    L = 32 * 32
    S = 24 * 24
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "loftr_pytorch/config/default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    local_feature_transformer = LocalFeatureTransformer(config["transformer"]["fine"]).to(device)

    n_embd = config["transformer"]["fine"]["n_embd"]
    x = torch.randn(B, L, n_embd).to(device)
    source = torch.randn(B, S, n_embd).to(device)
    x_mask = torch.rand(B, L) > 0.1
    source_mask = torch.rand(B, S) > 0.1
    feat0, feat1 = local_feature_transformer(x, source, x_mask.to(device), source_mask.to(device))
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
