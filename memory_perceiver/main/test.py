import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process

def setup(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=size)

# Function to be run by each process
def train(rank, size):
    
    # Initialize process group
    #dist.init_process_group(backend='nccl', init_method='env://')
    setup(rank, size)

    """tensor_list = [torch.zeros(2, dtype=torch.int64).to(rank) for _ in range(2)]
    tensor = torch.arange(2, dtype=torch.int64) + 1 * rank
    tensor = tensor.to(rank)
    dist.all_gather(tensor_list, tensor)
    if rank == 0:
        print(tensor_list)"""
    
    group1 = dist.new_group([1])
    tensor_dst = torch.zeros(2, device=rank, dtype=float)        
    if rank == 0:
        tensor_list = [torch.tensor([1,1], device=rank, dtype=float)]
        tensor_list.insert(0, torch.tensor([0,0], device=rank, dtype=float))
        dist.scatter(tensor_dst, tensor_list, 0)
    else:
        dist.scatter(tensor_dst, None, 0)
    print("rank : ", rank, "  ", tensor_dst)


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
