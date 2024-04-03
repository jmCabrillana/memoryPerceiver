import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process

def setup(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=size)

def scatter_dummy_src(tensor_batch, rank, size):
    if rank == 0:
        tensor_list = list(torch.split(tensor_batch, size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        dummy_dst = torch.zeros_like(tensor_list[0], device=0)
        dist.scatter(dummy_dst, tensor_list, 0)
    if rank > 0:
        dist.scatter(tensor_batch, None, 0)

def gather_dummy_src(tensor, rank, size):
    #  gather and batch 1, ..., size-1 to 0
    if rank > 0:
        dist.gather(tensor, None, 0)
    if rank == 0:
        tensor_list = list(torch.split(tensor, len(tensor)//size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        dummy_src = torch.zeros_like(tensor_list[0], device=0)
        dist.gather(dummy_src, tensor_list, 0)
        tensor_list.pop(0)
        tensor = torch.stack(tensor_list)

# Function to be run by each process
def train(rank, size):
    
    # Initialize process group
    #dist.init_process_group(backend='nccl', init_method='env://')
    setup(rank, size)
    
    """
    group1 = dist.new_group([1])
    tensor_dst = torch.zeros(2, device=rank, dtype=float)        
    if rank == 0:
        tensor_list = [torch.tensor([1,1], device=rank, dtype=float)]
        tensor_list.insert(0, torch.tensor([0,0], device=rank, dtype=float))
        print(torch.stack(tensor_list).tolist())
        dist.scatter(tensor_dst, tensor_list, 0)
    else:
        dist.scatter(tensor_dst, None, 0)
    print("rank : ", rank, "  ", tensor_dst)
    """
    

    """
    tensor_batch = torch.tensor([[4,7,8]], device=rank, dtype=float)
    if rank > 0:
        tensor_batch *= 0
    scatter_dummy_src(tensor_batch, rank, size-1)
    print("rank : ", rank, "  ", tensor_batch)
    """
    
    if rank > 0:
        tensor = torch.tensor([[1,2,1],[4,4,4]], device=rank, dtype=float)
        tensor *= rank
    if rank == 0:
        tensor = torch.zeros(6, 3, device=rank, dtype=float)
    gather_dummy_src(tensor, rank, size-1)
    print("rank : ", rank, "  ", tensor)
    

    # Clean up process group
    dist.destroy_process_group()
    

if __name__ == '__main__':
    # Number of processes (assumed to be equal to the number of GPUs)
    world_size = torch.cuda.device_count()

    # Spawn processes
    processes = []
    for rank in range(world_size):
        p = Process(target=train, args=(rank, world_size))
        p.start()
        processes.append(p)

    # Join processes
    for p in processes:
        p.join()
