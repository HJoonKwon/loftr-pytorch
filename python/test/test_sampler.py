import pytest
import torch
from torch.utils.data import Dataset, ConcatDataset
from loftr_pytorch.datasets.sampler import (
    RandomConcatSampler,
    DistributedPreSampledSampler,
)


class SimpleDataset(Dataset):
    def __init__(self, size, offset=0):
        self.size = size
        self.offset = offset  # To differentiate items from different subsets

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Return a simple tuple that includes the offset
        # This helps to distinguish elements from different subsets
        return (index + self.offset,)


# Create several SimpleDatasets with different sizes
datasets = [
    SimpleDataset(size=100, offset=0),
    SimpleDataset(
        size=150, offset=100
    ),  # Note: offsets are arbitrary, for distinguishing purposes
    SimpleDataset(size=50, offset=250),
]

# Concatenate datasets
concat_dataset = ConcatDataset(datasets)

# Set random seed for reproducibility
seed = 42


@pytest.mark.parametrize("world_size", [2, 4])
def test_sampler_sync_with_concat_dataset(world_size):

    # Placeholder to store pre_sampled_indices from each rank for comparison
    all_rank_indices = []

    for rank in range(world_size):
        # Ensure reproducibility
        torch.manual_seed(seed)

        # Simulate sampling pre_sampled_indices as you would at the beginning of each epoch
        random_concat_sampler = RandomConcatSampler(
            concat_dataset, n_samples_per_subset=10, seed=seed
        )
        pre_sampled_indices = list(iter(random_concat_sampler))

        # Store pre_sampled_indices for comparison
        all_rank_indices.append(pre_sampled_indices)

    # Check if all ranks have identical pre_sampled_indices
    for rank_indices in all_rank_indices[1:]:
        assert (
            rank_indices == all_rank_indices[0]
        ), "Pre-sampled indices are not synchronized across ranks."


@pytest.mark.parametrize("pad", [True, False])
def test_sampler_distribution(pad):
    world_size = 4  # Number of processes, e.g., 4 for this test
    indices_lengths = []
    # Create a sampler and dataloader for each process (simulate DDP setup)
    for rank in range(world_size):

        # Ensure reproducibility
        torch.manual_seed(seed)

        # Simulate sampling pre_sampled_indices as you would at the beginning of each epoch
        random_concat_sampler = RandomConcatSampler(
            concat_dataset, n_samples_per_subset=10, seed=seed
        )
        pre_sampled_indices = list(iter(random_concat_sampler))

        sampler = DistributedPreSampledSampler(
            dataset=concat_dataset,
            pre_sampled_indices=pre_sampled_indices,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Disable shuffle for testing predictability
            pad=pad,
            seed=seed,
        )

        # Collect all indices for this rank
        indices_for_rank = list(iter(sampler))
        indices_lengths.append(len(indices_for_rank))

        # With pad=False, ensure no repetition; with pad=True, ensure correct distribution
        if not pad:
            # Ensure coverage and possibly accept repetitions
            assert len(indices_for_rank) == len(
                set(indices_for_rank)
            ), "Indices should not repeat when pad=False."
            assert len(indices_for_rank) == len(
                sampler
            ), "Length of indices should match the sampler reported length."

    if pad:
        assert all(
            length == max(indices_lengths) for length in indices_lengths
        ), "All ranks should have the same number of samples when pad=True."
    else:
        assert sum(indices_lengths) == len(
            pre_sampled_indices
        ), "All samples should be covered when pad=False."
