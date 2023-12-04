import torch
import numpy as np
from einops import rearrange, repeat

def buffer_padding(buffer, dim, default_value=None, frames=32, numEnvs=1):
    
    # zero padding then transform: t b (d) -> b t (d)

    device = buffer.device
    pad0 = np.max([0, frames-len(buffer)])
    if default_value is None:
        padding = torch.zeros([pad0, numEnvs]+dim, device=device)
    else:
        padding = repeat(default_value, ' ... -> t0 b ...', t0=pad0, b=numEnvs).to(device)
    res = torch.swapaxes( 
            torch.cat([ padding, buffer ])
            , 0, 1)
    return res

def update_buffer(buff, key, element, frames=32):
    buff[key].append(element)
    if(len(buff[key])>frames):
        buff[key].pop(0)

def update_buffers(buff, dict, frames=32):
    for key, element in dict:
        update_buffer(buff, key, element, frames)


def calculate_returns(mem_rewards, death_index, gamma=0.99):
    sim_duration, numEnvs = mem_rewards.size()
    returns_estimates = torch.zeros_like(mem_rewards)
    for env_idx in range(numEnvs):
        sum = 0.0
        for t in reversed(range(sim_duration)):
            # If the agent is reset, set to 0
            if t in death_index[env_idx]:
                sum = 0

            # Reverse calcualtion of returns
            sum = gamma*sum + mem_rewards[t, env_idx]
            returns_estimates[t, env_idx] = sum
        
    return returns_estimates


def calculate_gae(mem_rewards, mem_values, death_index, gamma=0.99, gae_lambda=0.95):
    sim_duration, numEnvs = mem_rewards.size()
    gae_estimates = torch.zeros_like(mem_rewards)

    for env_idx in range(numEnvs):
        gae = 0.0 
        for t in reversed(range(sim_duration)):
            # If the agent is reset, set GAE to 0
            if t in death_index[env_idx]:
                gae = 0.0
                
            # Calculate the TD error and gae
            if t == sim_duration-1:
                delta_t = mem_values[t, env_idx]  
            else:
                delta_t= mem_rewards[t, env_idx] + gamma * mem_values[t+1, env_idx] - mem_values[t, env_idx]
            gae_estimates[t, env_idx] = delta_t + gamma * gae_lambda * gae
        
    return gae_estimates

def update_lr(current_lr, kl_div, kl_threshold=0.024, min_lr=1e-6, max_lr=1e-2):
    lr = current_lr
    if kl_div > (2.0 * kl_threshold):
        lr = max(current_lr / 1.5, min_lr)
    if kl_div < (0.5 * kl_threshold):
        lr = min(current_lr * 1.5, max_lr)
    return lr    