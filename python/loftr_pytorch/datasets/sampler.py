import torch
from torch.utils.data import Sampler, ConcatDataset
import torch.distributed as dist


class DistributedPreSampledSampler(Sampler):
    def __init__(
        self,
        dataset,
        pre_sampled_indices,
        num_replicas,
        rank,
        shuffle=True,
        seed=0,
        pad=True,
    ):
        """
        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from
            pre_sampled_indices (list): List of indices to sample from
            num_replicas (int, optional): Number of processes participating in distributed training
            rank (int, optional): Rank of the current process
            shuffle (bool, optional): Whether to shuffle indices or not
            seed (int, optional): Seed for shuffling
            pad (bool, optional): Whether to pad the dataset to ensure each process gets the same number of samples

        Ref:
            https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
            https://github.com/mlperf/training_results_v0.6/blob/master/NVIDIA/benchmarks/ssd/implementations/pytorch/sampler.py
            https://github.com/pytorch/pytorch/issues/25162
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.pre_sampled_indices = pre_sampled_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.pad = pad

        if self.pad:
            # Calculate the total size by rounding up to ensure each replica has the same number of samples
            self.total_size = (
                -(-len(self.pre_sampled_indices) // self.num_replicas)
                * self.num_replicas
            )
            # (Padding) Extend the pre_sampled_indices to match the total size if necessary
            self.pre_sampled_indices += self.pre_sampled_indices[
                : self.total_size - len(self.pre_sampled_indices)
            ]
        else:
            # No padding; the total size is just the length of pre_sampled_indices
            # It will be used for validation set to ensure each sample is evaluated
            self.total_size = len(self.pre_sampled_indices)

        assert self.total_size == len(self.pre_sampled_indices)

        # Calculate the actual number of samples each process should get
        # If padding is disabled, the number of samples can be different for each process
        self.num_samples = (
            self.total_size // self.num_replicas
            if self.pad
            else len(self.pre_sampled_indices[self.rank :: self.num_replicas])
        )

    def __iter__(self):
        # Generating a seed for shuffling using epoch and external seed
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        if self.pad:
            # Distributing indices with padding
            indices = self.pre_sampled_indices[
                self.rank : self.total_size : self.num_replicas
            ]
        else:
            # Distributing indices without padding, different batch sizes for each GPU are allowed
            indices = self.pre_sampled_indices[self.rank :: self.num_replicas]

        assert len(indices) == self.num_samples

        if self.shuffle:
            # Shuffling indices for this replica with epoch and external seed
            indices = torch.tensor(indices, dtype=torch.long)
            indices = indices[torch.randperm(len(indices), generator=g)].tolist()

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


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
