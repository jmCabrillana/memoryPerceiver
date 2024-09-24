import torch
import numpy as np
import os
import random
from einops import repeat, rearrange
import re
import shutil
from datetime import datetime

def init_history(history_keys, sim_duration, num_envs, action_dim, obs_dim, device):
    history = { key: torch.zeros(sim_duration, num_envs, device=device) for key in history_keys}
    history['obs'] = torch.zeros(sim_duration, num_envs, obs_dim, device=device)
    history['actions'] = torch.zeros(sim_duration, num_envs, action_dim, device=device)
    history['generator'] = torch.zeros(sim_duration, torch.cuda.device_count(), 16, dtype=torch.uint8) # must be on cpu
    return history

def buffer_padding(buffer, dim, default_value=None, frames=32, num_envs=1):
    
    # zero padding then transform: t b (d) -> b t (d)

    device = buffer.device
    pad0 = np.max([0, frames-len(buffer)])
    if default_value is None:
        padding = torch.zeros([pad0, num_envs]+dim, device=device)
    else:
        padding = repeat(default_value, '... -> t0 b ...', t0=pad0, b=num_envs).to(device)
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
    sim_duration, num_envs = mem_rewards.size()
    device = mem_rewards.device
    returns_estimates = torch.zeros(sim_duration, num_envs, device=device)
    sum = torch.ones(num_envs, device=device)
    
    for t in reversed(range(sim_duration)):
        # If the agent is reset, set to 0
        sum *= 1-death_index[t, :]

        # Reverse calcualtion of returns
        sum = gamma*sum + mem_rewards[t, :]
        returns_estimates[t, :] = sum
        
    return returns_estimates


def calculate_gae(mem_rewards, mem_values, next_values, death_index, gamma=0.99, gae_lambda=0.95):
    device = mem_rewards.device
    sim_duration, num_envs = mem_rewards.size()
    gae_estimates = torch.zeros(sim_duration, num_envs, device=device)
    gae = torch.ones(num_envs, device=device)
    
    for t in reversed(range(sim_duration)):
        # If the agent is reset, set GAE to 0
        gae *= 1-death_index[t, :]

        # Calculate the TD error and gae
        tmp_values = next_values if (t == sim_duration-1) else mem_values[t+1, :]
        delta_t = mem_rewards[t, :] + gamma * tmp_values - mem_values[t, :]
        gae = delta_t + gamma * gae_lambda * gae
        gae_estimates[t, :] = gae
        
    return gae_estimates

def create_mask(birth_time, sim_step, frames=32):
    device = birth_time.device
    num_envs = birth_time.size()[0]
    past_horizon=frames - (sim_step-birth_time+1)
    mask = repeat(torch.arange(frames, device=device), '... -> b ...', b=num_envs) < repeat(past_horizon, '... -> ... t', t=frames)
    return mask

def update_lr(current_lr, kl_div, kl_threshold=0.008, min_lr=1e-5, max_lr=1e-4):
    lr = current_lr
    if kl_div > (2.0 * kl_threshold):
        lr = current_lr / 1.5
    if kl_div < (0.5 * kl_threshold):
        lr = current_lr * 1.5
    lr = min(max(lr, min_lr), max_lr)
    return lr    

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_run(path_config, path_experiment):
    experiment_number = 0
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
        shutil.copy(path_config, path_experiment+'configuration_0.py')
    else:
        pattern = re.compile(r'configuration_(\d+)\.py')
        max_file = None

        # List all files in the specified directory
        for filename in os.listdir(path_experiment):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                if number >= experiment_number:
                    experiment_number = number
                    max_file = filename

        if max_file is None:
            shutil.copy(path_config, path_experiment+'configuration_0.py')
        else:
            path_last_experiment = path_experiment+max_file
            with open(path_config, 'rb') as f1, open(path_last_experiment, 'rb') as f2:
                if f1.read() != f2.read():
                    experiment_number += 1
                    shutil.copy(path_config, path_experiment+'configuration_'+str(experiment_number)+'.py')

    time_experiment = datetime.now().strftime("%m-%d_%H:%M:%S")
    path_run = path_experiment + 'run_' + str(experiment_number) +'_'+time_experiment+'/'
    return path_run