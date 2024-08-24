import os
import torch
import torch.distributed as dist

def setup_distributed(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=size)

def average_gradients(model, size, group):
    for param in model.parameters():
        if param.grad is not None: # !!! Temporary, audio not linked to loss for now (modality/projection['audio']) !!!
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
            param.grad.data /= size
            if dist.get_rank() !=  param.get_device():
                print(f"{dist.get_rank()} -> {param.device}")
                # sys.exit()

def scatter_dummy_src(tensor, rank, size):
    #  split batch at rank 0 to ranks  1, ..., size-1
    if rank == 0:
        tensor_list = list(torch.split(tensor, len(tensor)//size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        dst = torch.zeros_like(tensor_list[0], device=0)
    if rank != 0:
        tensor_list = None
        dst = tensor
    dist.barrier()
    dist.scatter(dst, tensor_list, src=0)

def gather_dummy_src(tensor, rank, size):
    #  gather and batch ranks 1, ..., size-1 to 0
    if rank != 0 :
        src = tensor
        tensor_list = None
    if rank == 0:
        tensor_list = list(torch.split(tensor, len(tensor)//size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        src = torch.zeros_like(tensor_list[0], device=0)
    dist.barrier()
    dist.gather(src, tensor_list, dst=0)
    if rank == 0:
        tensor_list.pop(0)
        tensor = torch.stack(tensor_list)