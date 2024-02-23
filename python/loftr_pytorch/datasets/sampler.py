import torch
from torch.utils.data import Sampler, ConcatDataset


class RandomConcatSampler(Sampler):
    """Random sampler for ConcatDataset. At each epoch, `n_samples_per_subset` samples will be draw from each subset
    in the ConcatDataset. If `subset_replacement` is ``True``, sampling within each subset will be done with replacement.
    However, it is impossible to sample data without replacement between epochs, unless bulding a stateful sampler lived along the entire training phase.

    For current implementation, the randomness of sampling is ensured no matter the sampler is recreated across epochs or not and call `torch.manual_seed()` or not.
    Args:
        shuffle (bool): shuffle the random sampled indices across all sub-datsets.
        repeat (int): repeatedly use the sampled indices multiple times for training.
            [arXiv:1902.05509, arXiv:1901.09335]
    NOTE: Don't re-initialize the sampler between epochs (will lead to repeated samples)
    NOTE: This sampler behaves differently with DistributedSampler.
          It assume the dataset is splitted across ranks instead of replicated.
    Ref:
        https://github.com/PyTorchLightning/pytorch-lightning/blob/e9846dd758cfb1500eb9dba2d86f6912eb487587/pytorch_lightning/trainer/training_loop.py#L373
        https://github.com/zju3dv/LoFTR/blob/master/src/datasets/sampler.py
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        n_samples_per_subset: int,
        subset_replacement: bool = True,
        shuffle: bool = True,
        repeat: int = 1,
        seed: int = None,
    ):
        assert isinstance(
            dataset, ConcatDataset
        ), "dataset should be torch.utils.data.ConcatDataset"
        assert repeat >= 1

        self.dataset = dataset
        self.n_subset = len(self.dataset.datasets)
        self.n_samples_per_subset = n_samples_per_subset
        self.n_samples = self.n_subset * self.n_samples_per_subset * repeat
        self.subset_replacement = subset_replacement
        self.repeat = repeat
        self.shuffle = shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        indices = []
        # sample from each sub-dataset
        for d_idx in range(self.n_subset):
            low = 0 if d_idx == 0 else self.dataset.cumulative_sizes[d_idx - 1]
            high = self.dataset.cumulative_sizes[d_idx]
            if self.subset_replacement:
                rand_tensor = torch.randint(
                    low,
                    high,
                    (self.n_samples_per_subset,),
                    generator=self.generator,
                    dtype=torch.long,
                )
            else:
                raise NotImplementedError
            indices.append(rand_tensor)
        indices = torch.cat(indices)  # (n_subset * n_samples_per_subset,)
        if self.shuffle:  # shuffle the sampled dataset (from multiple subsets)
            indices = self._random_permute(indices)

        # repeat the sampled indices (can be used for RepeatAugmentation or pure RepeatSampling)
        if self.repeat > 1:
            repeat_indices = [indices.clone() for _ in range(self.repeat - 1)]
            if self.shuffle:
                repeat_indices = self._random_permute(repeat_indices)
            indices = torch.cat([indices, *repeat_indices], 0)

        assert indices.shape[0] == self.n_samples
        return iter(indices.tolist())

    def _random_permute(self, x):
        return x[torch.randperm(len(x), generator=self.generator)]
