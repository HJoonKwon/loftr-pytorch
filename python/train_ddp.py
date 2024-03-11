import torch
import torch.optim as optim
import yaml, os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from loftr_pytorch.datasets.megadepth import (
    load_megadepth_dataloader,
    load_concatenated_megadepth,
)
from loftr_pytorch.datasets.sampler import (
    RandomConcatSampler,
    DistributedPreSampledSampler,
)
from loftr_pytorch.model.loftr import LoFTR
from loftr_pytorch.supervision.ground_truth import spvs_coarse, spvs_fine
from loftr_pytorch.loss.loftr_loss import LoFTRLoss


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "LOCALHOST"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def move_dict_to_cuda(tensor_dict, device):
    """
    Moves all tensors in a dictionary to CUDA device if CUDA is available.

    Args:
        tensor_dict (dict): A dictionary where values are PyTorch tensors.

    Returns:
        dict: A new dictionary with all tensors moved to CUDA device.
    """

    # Move each tensor in the dictionary to the selected device
    # Iterate through the dictionary, moving tensors to the device and leaving other types unchanged
    cuda_dict = {
        key: (
            value.to(device)
            if isinstance(value, torch.Tensor) and ("depth" not in key)
            else value
        )
        for key, value in tensor_dict.items()
    }

    return cuda_dict


def load_train_objs(base_path, lr=1e-4):
    data_root = os.path.join(base_path, "megadepth_indices")
    npz_root = os.path.join(data_root, "scene_info_0.1_0.7")
    train_list_path = os.path.join(data_root, "trainvaltest_list/train_list.txt")
    val_list_path = os.path.join(data_root, "trainvaltest_list/val_list.txt")
    config_path = os.path.join(
        "../python/loftr_pytorch/config/default.yaml",
    )
    config = yaml.safe_load(open(config_path, "r"))
    with open(train_list_path, "r") as f:
        train_npz_names = [f"{name.split()[0]}.npz" for name in f.readlines()]
    with open(val_list_path, "r") as f:
        val_npz_names = [f"{name.split()[0]}.npz" for name in f.readlines()]
    train_npz_paths = [
        os.path.join(npz_root, npz_name) for npz_name in train_npz_names[:2]
    ]
    val_npz_paths = [os.path.join(npz_root, npz_name) for npz_name in val_npz_names[:2]]

    train_dataset = load_concatenated_megadepth(
        base_path, train_npz_paths, "train", config["dataset"]["megadepth"]["train"]
    )
    val_dataset = load_concatenated_megadepth(
        base_path, val_npz_paths, "val", config["dataset"]["megadepth"]["val"]
    )

    model = LoFTR(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_module = LoFTRLoss(config)

    return train_dataset, val_dataset, model, optimizer, loss_module, config


# Ref: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_module: torch.nn.Module,
        train_data: Dataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        world_size: int,
        batch_size: int,
        save_every: int,
        config: dict,
    ) -> None:
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.model = model.to(gpu_id)
        self.loss_module = loss_module.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.config = config
        self.batch_size = batch_size

    def _run_batch(self, data, step):
        coarse_gt = spvs_coarse(data, self.config)
        data = move_dict_to_cuda(data, self.gpu_id)
        coarse_gt = move_dict_to_cuda(coarse_gt, self.gpu_id)

        result = self.model(data, coarse_gt)
        coarse_prediction, fine_prediction = result["coarse"], result["fine"]
        fine_gt = spvs_fine(data, coarse_gt, coarse_prediction, self.config)
        loss_dict = self.loss_module(
            data, coarse_prediction, fine_prediction, coarse_gt, fine_gt
        )
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()
        print(
            f"GPU ID {self.gpu_id}: Step {step}: loss = {loss_dict['loss'].item()}, coarse_loss = {loss_dict['coarse_loss']}, fine_loss = {loss_dict['fine_loss']}"
        )

    def _run_epoch(self, epoch):
        # Re-sample indices at the beginning of each epoch
        random_concat_sampler = RandomConcatSampler(
            self.train_data, **self.config["trainer"][self.config["trainer"]["sampler"]]
        )
        random_concat_sampler.set_epoch(epoch)
        pre_sampled_indices = list(iter(random_concat_sampler))

        # Creating a new instance of DistributedPreSampledSampler with the freshly sampled indices
        ddp_sampler = DistributedPreSampledSampler(
            self.train_data,
            pre_sampled_indices,
            num_replicas=self.world_size,
            rank=self.gpu_id,
            shuffle=True,
        )

        data_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, sampler=ddp_sampler
        )

        # Set epoch for the sampler to ensure proper shuffling
        data_loader.sampler.set_epoch(epoch)

        for step, data in enumerate(data_loader):
            print(
                f"GPU{self.gpu_id}: Epoch: {epoch}: step {step}/{len(data_loader)}, batch size {data['image0'].shape[0]}"
            )
            self.model.train()
            self._run_batch(data, step)
            dist.barrier()
            torch.cuda.empty_cache()

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            dist.barrier()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def train(
    rank,
    world_size,
    dataset,
    model,
    optimizer,
    loss_module,
    config,
    batch_size,
    total_epochs,
    save_every,
):
    ddp_setup(rank, world_size)
    # try/except for the graceful shutdown. Remove it if you want to debug the error
    try:
        trainer = Trainer(
            model,
            loss_module,
            dataset,
            optimizer,
            rank,
            world_size,
            save_every=save_every,
            config=config,
            batch_size=batch_size,
        )
        trainer.train(total_epochs)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, cleaning up...")
    finally:
        destroy_process_group()
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "base_path",
        default="/media/data1/megadepth",
        type=str,
        help="Path to the MegaDepth dataset",
    )
    parser.add_argument(
        "total_epochs", default=5, type=int, help="Total epochs to train the model"
    )
    parser.add_argument(
        "save_every", default=1, type=int, help="How often to save a snapshot"
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Input batch size on each device (default: 2)",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Initial learning rate (default: 1e-4)",
    )
    args = parser.parse_args()

    train_dataset, val_dataset, model, optimizer, loss_module, config = load_train_objs(
        args.base_path, lr=args.lr
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    mp.spawn(
        train,
        args=(
            world_size,
            train_dataset,
            model,
            optimizer,
            loss_module,
            config,
            args.batch_size,
            args.total_epochs,
            args.save_every,
        ),
        nprocs=world_size,
        join=True,
    )
