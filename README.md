# loftr-pytorch
Reimplementation of LoFTR in PyTorch, end-to-end transformer model for image matching

## Installation
Pytorch >= 2.0.0 is required. The code is tested with PyTorch 2.1.2 and Python 3.10.13.

## Differences from the original LoFTR
- [x] The attention mechanism implemented in Pytorch is used instead of the original attention mechanism. The recent one includes Flash Attention and Memory Attention, which are known to be more faster and efficient.
- [ ] The original Resnet block for downsampling is using convolutions 1x1 with stride 2, which might have caused the loss of information. The current implementation uses convolutions 3x3 with stride 2.
- [x] The original position embedding seems wrong refering to the DETR paper and its implementation. The current implementation uses the same position embedding as DETR, while keeping the dimension handling of the original LoFTR.
- [ ] The original LoFTR's transformer encoder uses Post-LayerNorm, while the current implementation uses Pre-LayerNorm, which is known to be more stable and faster from the recent literature.
- [ ] The original LoFTR's transformer encoder uses concatenation in the multi-head attention, while the current implementation uses addition.
