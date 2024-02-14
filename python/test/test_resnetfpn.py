import yaml 
import torch
import os
from loftr_pytorch.model.backbone import ResNetFPN_8_2 


def test_ResNetFPN_8_2():
    file_path = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(file_path, 'loftr_pytorch/config/default.yaml') 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = ResNetFPN_8_2(config['backbone'])

    x = torch.rand(1, 1, 256, 256)
    feat_c, feat_f = model(x) 
    block_dims = config['backbone']['block_dims']
    resolution = config['backbone']['resolution']
    feat_c_shape = [1, block_dims[-1], 256 // resolution[0], 256 // resolution[0]]
    feat_f_shape = [1, block_dims[0], 256 // resolution[1], 256 // resolution[1]]
    assert feat_c.shape == torch.Size(feat_c_shape)
    assert feat_f.shape == torch.Size(feat_f_shape)

    

