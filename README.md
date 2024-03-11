# loftr-pytorch
Reimplementation of LoFTR in PyTorch, end-to-end transformer model for image matching

## Installation
Pytorch >= 2.2.0 is required. The code is tested with PyTorch 2.2.0 and Python 3.10.13.

## Unit test 
```CUDA_VISIBLE_DEVICES``` is required to test torch.compile().
```bash
$ cd scripts
$ CUDA_VISIBLE_DEVICES=X bash test.sh
```

## DDP Training
**NOTE**: If zero-loss is printed, it means the given batch has no valid matches. This is not a bug, but a design to prevent DDP deadlock. The model is not updated because gradients are zero in this case. 
```bash
$ cd python
$ python train_ddp.py --help
usage: train_ddp.py [-h] [--batch_size BATCH_SIZE] [--lr LR] base_path total_epochs save_every

simple distributed training job

positional arguments:
  base_path             Path to the MegaDepth dataset
  total_epochs          Total epochs to train the model
  save_every            How often to save a snapshot

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Input batch size on each device (default: 2)
  --lr LR               Initial learning rate (default: 1e-4)
```

## Differences from the original LoFTR
- [ ] Remove pytorch lightning dependency for training.
- [x] Make the LoFTR model compilable to accelerate both inference and training speed.
- [x] Customized distributed sampler to enable DDP training.
- [x] Instead of einops, torch native functions are used for dealing with dimensions.
- [x] Enable Flash Attention and Memory Attention, which are known to be faster and more efficient.
- [x] Fix Position Embedding based on DETR's one. 
- [x] Add unit tests for improving readability and understanding.
- [ ] border_rm is constant in the original LoFTR, but it should change according to the window size..(?). No problem with border_rm = 2 when window_size = 5 because padding is 2 for fine preprocessing.
- [ ] The original LoFTR's transformer encoder uses Post-LayerNorm, while the current implementation uses Pre-LayerNorm, which is known to be more stable and faster from the recent literature.
- [ ] The original LoFTR's transformer encoder uses concatenation in the multi-head attention, while the current implementation uses addition.
- [ ] The original Resnet block for downsampling is using convolutions 1x1 with stride 2, which might have caused the loss of information. The current implementation uses convolutions 3x3 with stride 2.