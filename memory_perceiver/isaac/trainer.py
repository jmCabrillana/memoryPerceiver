import torch
import numpy as np
from einops import repeat, rearrange

def init_history(history_keys, sim_duration, numEnvs, action_dim, device):
    history = { key: torch.zeros(sim_duration, numEnvs, device=device) for key in history_keys}
    history['actions'] = torch.zeros(sim_duration, numEnvs, action_dim, device=device)
    history['generator'] = torch.zeros(sim_duration, torch.cuda.device_count(), 816, dtype=torch.uint8) # must be on cpu
    return history

def buffer_padding(buffer, dim, default_value=None, frames=32, numEnvs=1):
    
    # zero padding then transform: t b (d) -> b t (d)

    device = buffer.device
    pad0 = np.max([0, frames-len(buffer)])
    if default_value is None:
        padding = torch.zeros([pad0, numEnvs]+dim, device=device)
    else:
        padding = repeat(default_value, '... -> t0 b ...', t0=pad0, b=numEnvs).to(device)
    res = torch.cat([ padding, buffer ])
    res = rearrange(res, 't b ... -> b t ...')
    return res

def update_buffer(buff, key, element, frames=32):
    buff[key].append(element)
    if(len(buff[key])>frames):
        buff[key].pop(0)

def update_buffers(buff, dict, frames=32):
    for key, element in dict.items():
        update_buffer(buff, key, element, frames)


def calculate_returns(mem_rewards, death_index, gamma=0.99):
    sim_duration, numEnvs = mem_rewards.size()
    returns_estimates = torch.zeros_like(mem_rewards)
    sum = torch.zeros_like(mem_rewards[0])
    
    for t in reversed(range(sim_duration)):
        # If the agent is reset, set to 0
        sum *= 1-death_index[t, :]

        # Reverse calcualtion of returns
        sum = gamma*sum + mem_rewards[t, :]
        returns_estimates[t, :] = sum
        
    return returns_estimates


def calculate_gae(mem_rewards, mem_values, death_index, gamma=0.99, gae_lambda=0.95):
    sim_duration, numEnvs = mem_rewards.size()
    gae_estimates = torch.zeros_like(mem_rewards)
    gae = torch.zeros_like(mem_rewards[0])
    
    for t in reversed(range(sim_duration)):
        # If the agent is reset, set GAE to 0
        gae *= 1-death_index[t, :]

        # Calculate the TD error and gae
        if t == sim_duration-1:
            delta_t = mem_values[t, :]  
        else:
            delta_t= mem_rewards[t, :] + gamma * mem_values[t+1, :] - mem_values[t, :]
        gae_estimates[t, :] = delta_t + gamma * gae_lambda * gae
        
    return gae_estimates

def create_mask(birth_time, sim_step, frames=32):
        numEnvs = birth_time.size()[0]
        past_horizon=frames - (sim_step-birth_time+1)
        mask = repeat(torch.arange(frames), '... -> b ...', b=numEnvs) < repeat(past_horizon, '... -> ... t', t=frames)
        return mask

def update_lr(current_lr, kl_div, kl_threshold=0.024, min_lr=1e-6, max_lr=1e-2):
    lr = current_lr
    if kl_div > (2.0 * kl_threshold):
        lr = max(current_lr / 1.5, min_lr)
    if kl_div < (0.5 * kl_threshold):
        lr = min(current_lr * 1.5, max_lr)
    return lr    
    