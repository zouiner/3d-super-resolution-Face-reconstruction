import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.multiprocessing import spawn

def setup(rank, world_size):
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)

    # Create model
    model = torch.nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=10, num_workers=2, pin_memory=True)

    # Training loop
    dist.barrier()
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(rank), targets.to(rank)
            outputs = ddp_model(inputs)
            print(f"Rank {rank}, Epoch {epoch}, Outputs Mean: {outputs.mean().item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
