import torch
import numpy as np
import os
import random
from einops import repeat, rearrange
import re
import shutil
from datetime import datetime

def init_history(history_keys, sim_duration, num_envs, actions_dim, obs_dim, height, width, channels, device):
    history = {}
    for key in history_keys:
        if key == 'obs':
            history[key] = torch.zeros(sim_duration, num_envs, obs_dim, device=device)
        elif key == 'video':
            history[key] = torch.zeros(sim_duration, num_envs, height, width, channels, device=device)
        elif key.startswith('actions'):
            history[key] = torch.zeros(sim_duration, num_envs, actions_dim, device=device)
        elif key == 'generator':
            history[key] = torch.zeros(sim_duration, torch.cuda.device_count(), 16, dtype=torch.uint8) # must be on cpu
        else:
            history[key] = torch.zeros(sim_duration, num_envs, device=device)
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

def buffer_update(buff, key, element, frames=32):
    if isinstance(element, list):
        buff[key] += element
        while(len(buff[key])>frames):
            buff[key].pop(0)
    else:
        buff[key].append(element)
        if(len(buff[key])>frames):
            buff[key].pop(0)

def buffers_update(buff, dict, frames=32):
    for key, element in dict.items():
        buffer_update(buff, key, element, frames)

def buffers_pop():
    for key, element in dict.items():
        if element != []:
            element.pop()

    
def stack_buffer(buffer):
    out_dict = {}
    for key, element in buffer.items():
        if element != []:
            out_dict[key] = torch.stack(element).detach()
    return out_dict


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
        not_dones = 1-death_index[t, :]

        # Calculate the TD error and gae
        tmp_values = next_values if (t == sim_duration-1) else mem_values[t+1, :]
        delta_t = mem_rewards[t, :] + not_dones * gamma * tmp_values - mem_values[t, :]
        gae = delta_t + not_dones * gamma * gae_lambda * gae
        gae_estimates[t, :] = gae
        
    return gae_estimates

def calculate_grpo(r_grpo, rewards, birth_time, current_time):
    r_grpo_count = torch.max(torch.tensor(1, device=r_grpo.device), r_grpo[0])
    r_grpo_count_sum = max(1, r_grpo[0].sum())
    r_grpo_count_cumsum = torch.max(torch.tensor(1, device=r_grpo.device), r_grpo[0].cumsum(0))
    r_grpo_density = r_grpo[0] / r_grpo_count_sum
    r_grpo_mean = r_grpo[1].cumsum(0) / r_grpo_count_cumsum
    r_grpo_var = r_grpo[2].cumsum(0) / r_grpo_count_cumsum - r_grpo_mean ** 2 # torch.cumsum(r_grpo[1] ** 2)
    r_grpo_std = torch.sqrt(torch.max(torch.tensor(1e-8, device=r_grpo.device), r_grpo_var))
    sim_duration, num_env = rewards.shape
    times = torch.arange(current_time-sim_duration+1, current_time+1, device=r_grpo.device)
    times = repeat(times, 't -> t b', b=num_env)
    lifespan = (times - birth_time).to(torch.int)
    # coeff = (r_grpo_density != 0) * ( 1 / (1e-4 + r_grpo_density) )
    # weights = coeff *  (coeff != 0).sum() / coeff.sum()
    weights = r_grpo_density.cumsum(0)
    advantage = weights[lifespan] * (rewards - r_grpo_mean[lifespan])/r_grpo_std[lifespan]
    # print('lifespan', lifespan[-10: ,0])
    # print('rewards', rewards[-10: ,0])
    # print('density', r_grpo_density[lifespan][-10: ,0])
    # print('weights', weights[lifespan][-10: ,0])
    # print('r_grpo 0', r_grpo[0, lifespan][-10: ,0])
    # print('r_grpo 1', r_grpo[1, lifespan][-10: ,0])
    # print('r_grpo_mean', r_grpo_mean[lifespan][-10: ,0])
    # print('r_grpo_std', r_grpo_std[lifespan][-10: ,0])
    # print('advantage', advantage[-10: ,0])
    return advantage

def update_grpo(r_grpo, rewards, current_time, birth_time, max_count=1000):
    # Renormalize when above max_count:
    r_grpo_max = r_grpo[0].max()
    if r_grpo_max > max_count:
        # r_grpo[1, r_grpo[0] >= max_count] *= (max_count - 1) / r_grpo[0, r_grpo[0] >= max_count]
        # r_grpo[2, r_grpo[0] >= max_count] *= (max_count - 1) / r_grpo[0, r_grpo[0] >= max_count]
        # r_grpo[0, r_grpo[0] >= max_count] = max_count - 1
        r_grpo *= (max_count - 1) / r_grpo_max

    # Update
    lifespan = (current_time - birth_time).int()
    r_grpo[0, lifespan] += 1
    r_grpo[1, lifespan] += rewards
    r_grpo[2, lifespan] += rewards ** 2

    return r_grpo
    

def create_mask(birth_time, sim_step, frames=32):
    device = birth_time.device
    num_envs = birth_time.size()[0]
    past_horizon=frames - (sim_step-birth_time+1)
    mask = repeat(torch.arange(frames, device=device), '... -> b ...', b=num_envs) >= repeat(past_horizon, '... -> ... t', t=frames)
    return mask

def batch_and_mask_states(buffer, cf, mask, device):
    states={}
    mask = mask.logical_not()
    states['obs'] = buffer_padding(buffer['obs'], [cf.dim_observations], num_envs=cf.num_envs, frames=cf.frames)
    states['obs'][mask] *= 0
    if 'actions_star' in cf.inputs:
        states['actions_star'] = buffer_padding(buffer['actions_star'], [cf.dim_actions], num_envs=cf.num_envs, frames=cf.frames)
        states['actions_star'][mask] *= 0
    if 'video' in cf.inputs:
        states['video'] = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
        # states['video'] = tr.buffer_padding(torch.stack(buffer['video']), [cf.height, cf.width, 3], num_envs=cf.num_envs, frames=cf.frames)
        states['video'][mask] *= 0
    if 'audio' in cf.inputs:
        states['audio'] = torch.zeros(cf.num_envs, cf.len_audio_raw, device=device)
        # states['audio'] = tr.buffer_padding(torch.stack(buffer['video']), [cf.sample_audio_frame], num_envs=cf.num_envs, frames=cf.frames)
        states['audio'][mask] *= 0
    return states

def lr_kl_scheduler(current_lr, kl_div, kl_threshold=0.008, min_lr=1e-5, max_lr=1e-4):
    lr = current_lr
    if kl_div > (2.0 * kl_threshold):
        lr = current_lr / 1.5
    if kl_div < (0.5 * kl_threshold):
        lr = current_lr * 1.5
    lr = min(max(lr, min_lr), max_lr)
    return lr 

def lr_warm_up(step, min_lr=1e-5, max_lr=1e-4, steps_warm_up=1000):
    lr_tmp = min_lr + step*(max_lr-min_lr)/steps_warm_up
    return min(max_lr, lr_tmp)

def kl_scheduler_warm_up(step, current_lr, kl_div, kl_threshold=0.008, min_lr=1e-5, max_lr=1e-4, steps_warm_up=1000):
    lr = current_lr
    if kl_div > (2.0 * kl_threshold):
        lr = current_lr / 1.5
    if kl_div < (0.5 * kl_threshold) and step >= steps_warm_up:
        lr = current_lr * 1.5
    else:
        if step < steps_warm_up:
            lr = min_lr + step*(max_lr-min_lr)/steps_warm_up
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
    time_experiment = datetime.now().strftime("%m-%d_%H:%M:%S")
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
        shutil.copy(path_config, path_experiment / f'configuration_0_{time_experiment}.py')
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
            shutil.copy(path_config, path_experiment / f'configuration_0_{time_experiment}.py')
        else:
            path_last_experiment = path_experiment / max_file
            with open(path_config, 'rb') as f1, open(path_last_experiment, 'rb') as f2:
                if f1.read() != f2.read():
                    experiment_number += 1
                    shutil.copy(path_config, path_experiment / f'configuration_{experiment_number}_{time_experiment}.py')

    path_run = path_experiment / ('run_' + str(experiment_number) + '_' + time_experiment ) 
    return path_run

def print_detail_env(history):
    for key, value in history.items():
        if key not in ['actions', 'actions_star', 'actions_mean', 'actions_std', 'logprobs', 'generator', 'obs', 'rewards']:
            print(f"{key}: {value[:,0]}")
        if key in ['actions', 'actions_star', 'actions_mean', 'actions_std', 'obs']:
            print(f"{key}: {value[:,0, 0]}")

def print_detail_returns(history, unscaled_values, tmp_advantages, value_scaler):
    print('returns ', history['returns'][:,0])
    print('values mean', unscaled_values.mean())
    print('target mean', (tmp_advantages + unscaled_values).mean())
    print('scaler mean', value_scaler.running_mean)
    print('scaler var', value_scaler.running_variance)
    print('unscaled_values:', unscaled_values[:, 0])