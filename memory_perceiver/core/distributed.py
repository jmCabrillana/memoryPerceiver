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
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group("nccl", rank=rank, world_size=size)

def average_gradients(model, size, group, param_filter=''):
    for name, param in model.named_parameters():
        if param.grad is not None and name.startswith(param_filter): # !!! Temporary, audio not linked to loss for now (modality/projection['audio']) !!!
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
            param.grad.data /= max(1, size)
            if dist.get_rank() !=  param.get_device():
                print(f"{dist.get_rank()} -> {param.device}")
                # sys.exit()

def scatter_dummy_src(src_tensor, dst_tensor, rank, rank_agents, swap_axis=False):
    #  split batch at rank 0 to ranks  1, ..., size-1
    if rank_agents == [0]:
        dst_tensor.copy_(src_tensor)
        return
    if swap_axis:
        swap_eventually(src_tensor)
        swap_eventually(dst_tensor)
    if rank == 0:
        tensor_list = list(torch.split(src_tensor.contiguous(), len(src_tensor)//len(rank_agents)))
        if 0 not in rank_agents:
            tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
            dst_tensor = torch.zeros_like(tensor_list[0], device=0)
    if rank != 0:
        tensor_list = None
    dist.barrier()
    dist.scatter(dst_tensor, tensor_list, src=0)
    if rank in rank_agents:
        pass
        # dst_tensor.copy_(dst)
    if swap_axis:
        swap_eventually(src_tensor)
        swap_eventually(dst_tensor)
        

def gather_dummy_src(src_tensor, dst_tensor, rank, rank_agents, swap_axis=False):
    #  gather and batch agent ranks to sim_rank (usually 1, ..., size-1 to 0)
    if rank_agents == [0]:
        dst_tensor.copy_(src_tensor)
        return
    if swap_axis:
        swap_eventually(src_tensor)
        swap_eventually(dst_tensor)
    if rank != 0 :
        tensor_list = None
    if rank in rank_agents:
        src = src_tensor.contiguous()
    if rank == 0:
        tensor_list = list(torch.split(dst_tensor.contiguous(), len(dst_tensor)//len(rank_agents)))
        if 0 not in rank_agents:
            tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
            src = torch.zeros_like(tensor_list[0], device=0)
    dist.barrier()
    dist.gather(src, tensor_list, dst=0)
    if rank == 0:
        if 0 not in rank_agents:
            tensor_list.pop(0)
        dst_tensor.copy_(torch.cat(tensor_list, dim=0))
    if swap_axis:
        swap_eventually(src_tensor)
        swap_eventually(dst_tensor)

def swap_eventually(t):
    if t is not None:
        t.swapaxes_(0, 1)