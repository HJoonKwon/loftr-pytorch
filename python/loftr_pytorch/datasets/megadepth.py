import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import h5py
import os, copy
import cv2
from loftr_pytorch.supervision.utils import inverse_transformation
from loftr_pytorch.datasets.sampler import RandomConcatSampler


def get_new_size(size, long_dim=None, div_factor=None):
    w, h = size
    if long_dim is not None:  # resize the longer edge
        scale = long_dim / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h

    divisible_size = get_divisible_size((w_new, h_new), div_factor=div_factor)
    return divisible_size


def get_divisible_size(size, div_factor=None):
    w, h = size
    if div_factor is not None:
        w_new, h_new = map(lambda x: int((x // div_factor) * div_factor), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(img, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(
        img.shape[-2:]
    ), f"{pad_size} < {max(img.shape[-2:])}"
    assert img.ndim == 2
    mask = None
    padded = np.zeros((pad_size, pad_size), dtype=img.dtype)
    padded[: img.shape[0], : img.shape[1]] = img
    if ret_mask:
        mask = np.zeros((pad_size, pad_size), dtype=bool)
        mask[: img.shape[0], : img.shape[1]] = True
    return padded, mask


def load_concatenated_megadepth(base_path, npz_paths, mode, config):
    datasets = []
    for npz_path in npz_paths:
        dataset = MegaDepth(base_path, npz_path, mode, config)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def load_megadepth_dataloader(base_path, npz_paths, mode, config, **loader_params):
    """
    Args:
        base_path (str): path to the MegaDepth dataset
        npz_paths (list): list of paths to the npz files
        mode (str): "train", "val" or "test"
        config (dict): full configuration
        loader_params (dict): parameters for DataLoader

    Returns:
        DataLoader: DataLoader for MegaDepth dataset

    NOTE: loader_params inlcude sampler, batch_size, shuffle, num_workers, etc.
    """
    assert mode in ["train", "val", "test"], f"Invalid mode {mode}"
    dataset = load_concatenated_megadepth(
        base_path, npz_paths, mode, config["dataset"]["megadepth"][mode]
    )
    # TODO:: mode dependent sampler
    sampler = RandomConcatSampler(
        dataset, **config["trainer"][config["trainer"]["sampler"]]
    )
    return DataLoader(dataset, sampler=sampler, **loader_params)


class MegaDepth(Dataset):
    def __init__(
        self,
        base_path,
        npz_path,
        mode,
        config,
    ):
        super().__init__()

        assert mode in ["train", "val", "test"], f"Invalid mode {mode}"

        self.base_path = base_path
        self.mode = mode
        self.scene_id = npz_path.split(".")[0]
        self.long_dim = config["long_dim"]
        self.div_factor = config["div_factor"]
        self.coarse_scale = config["coarse_scale"]
        self.image_padding = config["image_padding"]
        self.depth_padding = config["depth_padding"]
        self.depth_max_size = 2000 if self.depth_padding else None
        self.min_overlap_score = config["min_overlap_score"]

        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = [
            copy.deepcopy(pair_info)
            for pair_info in self.scene_info["pair_infos"]
            if pair_info[1] > self.min_overlap_score
        ]

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
        img_path0 = os.path.join(self.base_path, self.scene_info["image_paths"][idx0])
        img_path1 = os.path.join(self.base_path, self.scene_info["image_paths"][idx1])
        img0, mask0, scale0 = self._read_gray(
            img_path0, self.long_dim, self.div_factor, self.image_padding
        )
        img1, mask1, scale1 = self._read_gray(
            img_path1, self.long_dim, self.div_factor, self.image_padding
        )

        if self.mode in ["train", "val"]:
            depth_path0 = os.path.join(
                self.base_path, self.scene_info["depth_paths"][idx0]
            )
            depth_path1 = os.path.join(
                self.base_path, self.scene_info["depth_paths"][idx1]
            )
            depth0 = self._read_depth(depth_path0, self.depth_max_size)
            depth1 = self._read_depth(depth_path1, self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        K0 = torch.from_numpy(self.scene_info["intrinsics"][idx0]).double()
        K1 = torch.from_numpy(self.scene_info["intrinsics"][idx1]).double()

        T0 = torch.from_numpy(self.scene_info["poses"][idx0]).double()
        T1 = torch.from_numpy(self.scene_info["poses"][idx1]).double()
        T_0to1 = T1 @ inverse_transformation(T0)
        T_1to0 = inverse_transformation(T_0to1)

        data = {
            "image0": img0,  # (1, h, w)
            "depth0": depth0,  # (h, w)
            "image1": img1,
            "depth1": depth1,
            "hw0_i": img0.shape[-2:],  # (h, w)
            "hw1_i": img1.shape[-2:],
            "T_0to1": T_0to1,  # (4, 4)
            "T_1to0": T_1to0,
            "K0": K0,  # (3, 3)
            "K1": K1,
            "scale0": scale0,  # [scale_w, scale_h]
            "scale1": scale1,
            "dataset_name": "MegaDepth",
            "scene_id": self.scene_id,
            "overlap_score": overlap_score,
            "pair_id": idx,
            "pair_names": (
                self.scene_info["image_paths"][idx0],
                self.scene_info["image_paths"][idx1],
            ),
        }

        if mask0 is not None:
            assert self.coarse_scale is not None
            mask0, mask1 = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=self.coarse_scale,
                mode="nearest",
                recompute_scale_factor=False,
            )[0].bool()
            data.update({"mask0": mask0, "mask1": mask1})

        return data

    def _read_gray(self, path, long_dim=None, div_factor=None, padding=False):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        old_size = img.shape[:2][::-1]
        new_size = get_new_size(old_size, long_dim, div_factor)
        img = cv2.resize(img, new_size)
        scale = torch.tensor(old_size, dtype=torch.float) / torch.tensor(
            new_size, dtype=torch.float
        )

        mask = None
        if padding:
            img, mask = pad_bottom_right(img, pad_size=long_dim, ret_mask=True)

        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)
        mask = torch.from_numpy(mask)
        return img, mask, scale

    def _read_depth(self, path, pad_size=None):
        depth = np.array(h5py.File(path, "r")["depth"])
        if pad_size is not None:
            depth, _ = pad_bottom_right(depth, pad_size, ret_mask=False)
        depth = torch.from_numpy(depth).float()
        return depth
