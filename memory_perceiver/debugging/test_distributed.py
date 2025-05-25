import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process
from einops import rearrange

def setup(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    print(f"nccl {rank}/{size}")
    dist.init_process_group("nccl", rank=rank, world_size=size) #, device_id=torch.device(f'cuda:{rank}'), init_method='env://'

def swap_unswap_arguments(f, *args, **kwargs):
    args = list(args)
    args[0] = rearrange(args[0], 't b ... -> b t ...').contiguous()
    f(*args, **kwargs)
    args[0] = rearrange(args[0], 'b t ... -> t b ...')

def scatter_dummy_src(tensor, rank, size, swap_axis=False):
    #  split batch at rank 0 to ranks  1, ..., size-1
    if swap_axis:
        tensor.swapaxes_(0, 1)
    if rank == 0:
        tensor_list = list(torch.split(tensor.contiguous(), len(tensor)//size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        dst = torch.zeros_like(tensor_list[0], device=0)
    if rank != 0:
        tensor_list = None
        dst = torch.zeros_like(tensor.contiguous())
    dist.barrier()
    dist.scatter(dst, tensor_list, src=0)
    if rank != 0:
        tensor.copy_(dst)
    if swap_axis:
        tensor.swapaxes_(0, 1)

def gather_dummy_src(tensor, rank, size, swap_axis=False):
    #  gather and batch ranks 1, ..., size-1 to 0
    if swap_axis:
        tensor.swapaxes_(0, 1)
    if rank != 0 :
        tensor_list = None
        src = tensor.contiguous()
    if rank == 0:
        tensor_list = list(torch.split(tensor.contiguous(), len(tensor)//size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        src = torch.zeros_like(tensor_list[0], device=0)
    dist.barrier()
    dist.gather(src, tensor_list, dst=0)
    if rank == 0:
        tensor_list.pop(0)
        tensor.copy_(torch.cat(tensor_list, dim=0))
    if swap_axis:
        tensor.swapaxes_(0, 1)

# Function to be run by each process
def train(rank, size):
    
    # Initialize process group
    # os.environ['RANK'] = f'{rank}'
    # os.environ['WORLD_SIZE'] = f'{size}'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group(backend='nccl', init_method='env://', device_id=torch.device(f'cuda:{rank}'))
    setup(rank, size)
    
    # group1 = dist.new_group([1])
    tensor = torch.ones(1, device=rank, dtype=float) * rank
    print(f"{rank}: {tensor}")
    # dist.barrier()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"{rank}: {tensor}")
    # tensor_dst = torch.zeros(2, device=rank, dtype=float)     
    # print("rank : ", rank, "  ", tensor_dst)   
    # if rank == 0:
    #     tensor_list = [torch.tensor([1,1], device=rank, dtype=float)]
    #     tensor_list.insert(0, torch.tensor([0,0], device=rank, dtype=float))
    #     print(torch.stack(tensor_list).tolist())
    #     dist.scatter(tensor_dst, tensor_list, 0)
    # else:
    #     dist.scatter(tensor_dst, None, 0)
    # print("rank : ", rank, "  ", tensor_dst)
    
    
    # if rank == 0:
    #     tensor_batch = torch.tensor([[1,1,2],
    #                                  [4,7,8],
    #                                  [2,2,3],
    #                                  [4,7,8],
    #                                  [3,3,4],
    #                                  [4,7,8],], device=rank, dtype=float)
    # if rank > 0:
    #     tensor_batch = torch.zeros(6,3, device=rank, dtype=float)
    # scatter_dummy_src(tensor_batch, rank, size-1, swap_axis=True)
    # print(f"rank : {rank}, , {tensor_batch}")
    
    
    # if rank > 0:
    #     tensor = torch.tensor([[1, 2],[4, 5], [7,8]], device=rank, dtype=float)
    #     tensor *= rank
    # if rank == 0:
    #     tensor = torch.zeros(3, 2, device=rank, dtype=float)
    # print("rank : ", rank, "  ", tensor)
    # # gather_dummy_src(tensor, rank, size-1, swap_axis=True)
    # torch.distributed.
    # print("rank : ", rank, "  ", tensor)

    # group_agents=dist.new_group(list(range(1,4)))
    # print(list(range(1,size)))
    # if rank > 0:
    #     tensor = torch.tensor([[1,2,1],[3,3,3]], device=rank, dtype=float)
    #     tensor *= rank
    # if rank == 0:
    #     tensor = torch.zeros(2, 3, device=rank, dtype=float)
    # if rank > 0:
    #     dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_agents)
    #     #dist.barrier(group_agents)
    # #dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # print(f"{rank}: {tensor}")

    # Clean up process group
    dist.barrier()
    dist.destroy_process_group()
    

if __name__ == '__main__':
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO" # INFO WARNING FATAL
    os.environ["NCCL_DEBUG"]="TRACE" # WARN INFO DEBUG TRACE
    os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL" # OFF INFO DETAIL
    os.environ["TORCH_NCCL_ENABLE_MONITORING"]="1" # 0 1
    os.environ["CUDA_LAUNCH_BLOCKING"]="1" # 0 1 

    # Number of processes (assumed to be equal to the number of GPUs)
    world_size = torch.cuda.device_count()
    print("world_size", world_size)

    # manager=torch.multiprocessing.Manager()
    # d=manager.dict()

    # Spawn processes
    print(torch.device("cuda:1"))
    processes = []
    for rank in range(world_size):
        p = Process(target=train, args=(rank, world_size))
        p.start()
        processes.append(p)

    # Join processes
    for p in processes:
        p.join()
