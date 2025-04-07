import os
import torch
import torch.distributed as dist
from einops import rearrange

def swap_unswap_arguments(f, *args, **kwargs):
    args = list(args)
    args[0] = rearrange(args[0], 't b ... -> b t ...').contiguous()
    out = f(*args, **kwargs)
    out = rearrange(out, 'b t ... -> t b ...')
    return out

def setup_distributed(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=size)

def average_gradients(model, size, group, param_filter=''):
    for name, param in model.named_parameters():
        if param.grad is not None and name.startswith(param_filter): # !!! Temporary, audio not linked to loss for now (modality/projection['audio']) !!!
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
            param.grad.data /= max(1, size)
            if dist.get_rank() !=  param.get_device():
                print(f"{dist.get_rank()} -> {param.device}")
                # sys.exit()

def scatter_dummy_src(tensor, rank, size, swap_axis=False):
    #  split batch at rank 0 to ranks  1, ..., size-1
    if size == 0:
        return
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
    if size == 0:
        return
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