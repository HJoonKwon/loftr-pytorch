import torch
import numpy as np
import os, yaml
from loftr_pytorch.datasets.megadepth import MegaDepth


def test_megadepth():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "loftr_pytorch/config/default.yaml",
    )
    config = yaml.safe_load(open(config_path, "r"))
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "../resources/megadepth_sample",
    )
    npz_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "../resources/megadepth_sample/sample.npz",
    )

    megadepth = MegaDepth(
        base_path, npz_path, "train", config["dataset"]["megadepth"]["train"]
    )

    data = megadepth[0]
