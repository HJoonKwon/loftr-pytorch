import torch
from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend

SDPBackend.__module__ = "torch.backends.cuda"
SDPAParams.__module__ = "torch.backends.cuda"

B = 4
L = 32 * 32
S = 24 * 24
n_embd = 32
n_heads = 4

num_devices = torch.cuda.device_count()
device = f"cuda:{num_devices-1}" if torch.cuda.is_available() else "cpu"


def test_flash_attention_enabled():
    q = torch.randn(B, n_heads, L, n_embd // n_heads).type(torch.float16).to(device)
    k = torch.randn(B, n_heads, S, n_embd // n_heads).type(torch.float16).to(device)
    v = torch.randn(B, n_heads, S, n_embd // n_heads).type(torch.float16).to(device)
    input = SDPAParams(q, k, v, None, 0.1, False)
    assert torch.backends.cuda.can_use_flash_attention(input)


def test_mem_efficient_attention_enabled():
    q = torch.randn(B, n_heads, L, n_embd // n_heads).to(device)
    k = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    v = torch.randn(B, n_heads, S, n_embd // n_heads).to(device)
    input = SDPAParams(q, k, v, None, 0.1, False)

    assert torch.backends.cuda.can_use_efficient_attention(input)
