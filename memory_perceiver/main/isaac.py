from os.path import join, dirname
import os
import sys
import gc
import logging
import argparse
from copy import deepcopy
import numpy as np
import torchshow as ts
import torchvision
from einops import rearrange

import torch
from torch import autocast
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append(join(dirname(dirname(os.getcwd()))))
sys.path.append(os.getcwd())

from omni.isaac.lab.app import AppLauncher
from memoryPerceiver.memory_perceiver.main.configuration import cf
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_memories, reset_memories
from memoryPerceiver.memory_perceiver.core.distributed import setup_distributed, average_gradients, scatter_dummy_src, gather_dummy_src
from memoryPerceiver.memory_perceiver.core.miscellaneous import custom_random, print_cuda_tensors
import memoryPerceiver.memory_perceiver.isaac.trainer as tr
from memoryPerceiver.memory_perceiver.isaac.scaler import RunningStandardScaler
from memoryPerceiver.memory_perceiver.isaac.plot import Plotter
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCritic, ActorCriticSimple, ActorCriticPerceiver, create_agent

def get_states(obs_stack, sim_step, cf, birth_time, device):
    states={}
    mask = tr.create_mask(birth_time, sim_step, frames=cf.frames)
    states['observations'] = tr.buffer_padding(obs_stack, [cf.dim_observations],
                                                num_envs=cf.num_envs, frames=cf.frames)
    states['observations'][mask] *= 0
    if cf.input_video:
        states['video'] = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
        # states['video'] = tr.buffer_padding(torch.stack(buffer['video']), [cf.height, cf.width, 3], num_envs=cf.num_envs, frames=cf.frames)
        states['video'][mask] *= 0
        states['video'] = rearrange(states['video'], 'b t h w c -> b c t h w')
    if cf.input_audio:
        states['audio'] = torch.zeros(cf.num_envs, cf.len_audio_raw, device=device)
    return states


def isaac_training(cf, rank, size):

    setup_distributed(rank, size)
    rank_agents = list(range(1, size))
    rank_sim = 0
    group_agents = dist.new_group(rank_agents)
    device = torch.device(rank)

    cf.num_envs = cf.num_envs//(size-1) if rank in rank_agents else cf.num_envs
    
    if rank == rank_sim:
        # !!! Load isaac modules after the variables
        # env = VecEnvRLGames(headless=cfg.headless, sim_device=rank_sim, enable_viewport=cfg.enable_viewport)
        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app
        import gymnasium as gym
        from omni.isaac.core.utils.torch.maths import set_seed
        import omni.usd
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.stage import open_stage, save_stage
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        enable_extension("omni.isaac.sensor")
        env_cfg = parse_env_cfg(
            cf.task, device=str(device), num_envs=cf.num_envs, use_fabric=False
        )
        env = gym.make(cf.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        seed_isaac = set_seed(cf.seed, torch_deterministic=False)
        env.reset()
        value_scaler = RunningStandardScaler(size=1, device=device)  
        if os.path.exists(cf.path_scaler):
            value_scaler.load_state_dict(torch.load(cf.path_scaler))  

    if rank in rank_agents:
        model = create_agent(cf, rank)
        model_student = create_agent(cf, rank)
        if isinstance(model, ActorCriticPerceiver):
            memories = create_memories(model.actor, cf.num_envs, device=device)
        else :
            memories = None
            # last_memories = torch.zeros_like(memories['cross_cmem'], device=device)
        optimizer = torch.optim.Adam(model_student.parameters(), lr=cf.learning_rate)
        lr=cf.learning_rate
        scaler = GradScaler()
        if rank == 1:
            plotter = Plotter('tensorboard', titles=['loss_actor', 'loss_critic', 'loss_entropy', 'actions_gap', 'returns'],
                                path_plots=cf.path_experiment)

    history_keys = ['obs', 'actions', 'values', 'logprobs', 'rewards', 'returns', 'returns_targets', 
                    'advantages', 'birth_time', 'death_index', 'generator']
    history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, cf.dim_observations, device)
    previous_history = deepcopy(history)
    buffer = {'video':[], 'audio':[], 'obs':[], 'rewards':[], 'actions_star':[]}
    obs = torch.zeros(cf.num_envs, cf.dim_observations, device=rank)
    video = torch.zeros(cf.num_envs, cf.height, cf.width, cf.channels, device=rank, dtype=torch.uint8)
    reward = torch.zeros(cf.num_envs, 1, device=rank)
    actions_star = torch.zeros(cf.frames_future, cf.num_envs, cf.dim_actions, device=rank)     
    sim_count, sim_step = -1, 0
    sim_running = torch.tensor(False, device=rank)
    world_playing = torch.tensor(False, device=rank)
    if rank == rank_sim:
        sim_running.fill_(simulation_app.is_running())
    dist.broadcast(sim_running, src=rank_sim)    
    
    while sim_running:
        if rank == rank_sim:
            world_playing.fill_(env.unwrapped.sim.is_playing())
            sim_running.fill_(simulation_app.is_running())
        dist.broadcast(world_playing, src=0)
        dist.broadcast(sim_running, src=0)
        if world_playing:
            if sim_step % cf.sim_duration == 0:

                ### Plots

                if rank == 1 and sim_count % (cf.num_replay+1) == 1 :
                    plotter.plot_all([('loss_actor', 64), ('loss_critic', 64), ('loss_entropy', 64), ('actions_gap', 64), 'returns'])

                ### Initialization

                sim_count += 1
                seed = sim_count // (cf.num_replay+1)
                is_inference =  sim_count % (cf.num_replay+1) == 0 # not is_inference
                dist.destroy_process_group(group_agents)
                group_agents = dist.new_group(rank_agents) # It is necessary to reset this group here to avoid duplicate gpu errors
                # if rank == rank_sim:
                #     env.unwrapped.sim.reset(soft=True) #False
                
                if rank == rank_sim:
                    if len(history['rewards'])>0:
                        print(f"$$$ r{rank} CUMUL REWARD: {history['rewards'].sum().item()} $$$")
                    print(f"SEED: {seed}")
                    print(f"SIM_COUNT: {sim_count}")

                if is_inference:
                    if rank in rank_agents:
                        # print(f"MEM:  {F.mse_loss(last_memories, memories['cross_cmem']).item()}")
                        if sim_count == 0 and os.path.exists(cf.path_model):
                            state_dict = torch.load(cf.path_model)
                            model.load_state_dict(state_dict['model'])
                            optimizer.load_state_dict(state_dict['optimizer'])
                        else:
                            if sim_count % ( (cf.num_replay+1)*cf.student_update_step ) == 0 :
                                model.load_state_dict(model_student.state_dict())
                        dist.barrier(group=group_agents)
                        if rank == 1:
                            torch.save({'model': model_student.state_dict(), 'optimizer': optimizer.state_dict()}, cf.path_model)
                        model.eval()
                        history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, cf.dim_observations, device)
                    if rank == rank_sim:
                        torch.save(value_scaler.state_dict(), cf.path_scaler)
                    full_video = []
                    for key in buffer.keys():
                        buffer[key]=[]
                else:
                    previous_history = deepcopy(history)
                    history['values'] = rearrange(history['values'], 't b -> b t').contiguous()
                    gather_dummy_src(history['values'], rank, size-1)
                    history['values'] = rearrange(history['values'], 'b t -> t b')
                    if rank == rank_sim:
                        unscaled_values = value_scaler(history['values'], inverse=True) 
                        print('unscaled_values:', unscaled_values[:, 0])
                        previous_history['returns'] = tr.calculate_returns(history['rewards'], history['death_index'], gamma=cf.gamma)
                        tmp_advantages = tr.calculate_gae(history['rewards'], unscaled_values, history['death_index'], 
                                                          gamma=cf.gamma, gae_lambda=cf.c_lambda)
                        previous_history['values'] = value_scaler(unscaled_values, train=True) # != history['values'] because update
                        previous_history['returns_targets'] = value_scaler(tmp_advantages + unscaled_values, train=True)
                        previous_history['advantages'] = (tmp_advantages - tmp_advantages.mean()) / (tmp_advantages.std() + 1e-8)  #dim=1 keepdim=True
                    for key in ['obs', 'rewards', 'death_index', 'returns', 'values', 'returns_targets', 'advantages']:
                        swap_batch_time_tmp = rearrange(previous_history[key], 't b ... -> b t ...').contiguous()
                        scatter_dummy_src(swap_batch_time_tmp, rank, size-1)
                        previous_history[key] = rearrange(swap_batch_time_tmp, 'b t ... -> t b ...')
                    
                    if rank == 1:
                        plotter.add('returns', previous_history['returns'][0, :].mean().item(), sim_count)
                        for key, value in previous_history.items():
                            if key not in ['actions', 'logprobs', 'generator', 'obs']:
                                print(f"{key}: {value[:,0]}")
                            if key == 'actions':
                                print(f"{key}: {value[:,0, 0]}")
                    if rank in rank_agents and isinstance(model, ActorCriticPerceiver):
                            tmp_memories = memories
                            memories = torch.load(cf.path_experiment+'memories.pt')
                            dist.barrier(group=group_agents)
                            if rank == 1:
                                torch.save(tmp_memories, cf.path_experiment+'memories.pt')
                            del tmp_memories
                    if rank == rank_sim:
                        print('returns mean', previous_history['returns'].mean())
                        print('values mean', unscaled_values.mean())
                        print('target mean', (tmp_advantages + unscaled_values).mean())
                        print('scaler mean', value_scaler.running_mean)
                        print('scaler var', value_scaler.running_variance)

                    backward_count= 0
                sim_step = 0

            if is_inference:
                
                ### Get agent observations 

                if rank == rank_sim:
                    all_obs = env.unwrapped._get_observations()
                    obs = all_obs['policy'].clone()    # (num_envs, d)
                    if cf.render:
                        video = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                        #ts.save(video[0], '/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/live.jpg')
                        full_video.append(video.cpu())
                    reward=env.unwrapped._get_rewards()    # (num_envs, 1)
                    history['rewards'][sim_step] = reward
                for x in [obs, reward]: #video
                    scatter_dummy_src(x, rank, size-1) 
                tr.update_buffers(buffer, {'obs': obs, 'video': video, 'rewards': reward}, frames=cf.frames)
                if cf.render:
                    scatter_dummy_src(video, rank, size-1)
                    tr.update_buffers(buffer, {'video': video}, frames=cf.frames)

                ### Record dones and birth index

                dones = torch.zeros(cf.num_envs, device=rank)
                if rank == rank_sim:
                    dones = env.unwrapped.last_dones # dones, _ = env.unwrapped._get_dones()
                    dones = dones.int().clone()
                    if sim_step >= 1:
                        history['birth_time'][sim_step] = history['birth_time'][sim_step-1]
                    history['birth_time'][sim_step, dones==1] = sim_step
                    history['death_index'][sim_step]=dones

                scatter_dummy_src(history['death_index'][sim_step], rank, size-1) 
                scatter_dummy_src(history['birth_time'][sim_step], rank, size-1) 

            ## reset memories

            if rank in rank_agents and isinstance(model, ActorCriticPerceiver):
                if is_inference :
                    dones_list = torch.nonzero(history['death_index'][sim_step].int()).flatten().tolist()
                else: 
                    dones_list = torch.nonzero(previous_history['death_index'][sim_step].int()).flatten().tolist()
                reset_memories(memories, indexes=dones_list)

            ### Agent
            
            if sim_step % cf.frames_future == 0 and rank in rank_agents:
                if is_inference:
                    ## Inference, save history and memory update

                    obs_stack = torch.stack(buffer['obs'])
                    states=get_states(obs_stack, sim_step, cf, history['birth_time'][sim_step], device)
                    history['generator'][sim_step, rank] = torch.cuda.get_rng_state(rank).clone() 
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        actions, values, logprobs, memories = model(states, memories)   
                    history['actions'][sim_step] = actions  
                    history['logprobs'][sim_step] = logprobs 
                    history['values'][sim_step:sim_step+cf.frames_future] = values  # for cumulative sum  

                    # Instanciate action before dispatch
                    actions_star = model.instanciate_actions(actions, dt=cf.frames_future).to(torch.float32) # important for gather !!!!

                else:

                    ## Process inputs
                    sim_step_rand = torch.randint(0, int(1*cf.sim_duration), (1,)).item()   
                    # sim_step_rand=sim_step 
                    actions, logprobs, values, birth_time = [previous_history[key][sim_step_rand].to(device) \
                                                for key in ['actions', 'logprobs', 'values', 'birth_time']]  
                    obs_stack = previous_history['obs'][sim_step_rand-cf.frames+1:sim_step_rand+1]
                    states=get_states(obs_stack, sim_step_rand, cf, birth_time, device)

                    perform_update = custom_random(sim_count, sim_step_rand) < cf.update_probability and \
                                        sim_step_rand <= cf.sim_duration-1
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        if perform_update :
                            returns_targets = previous_history['returns_targets'][sim_step_rand].to(device)
                            advantages = previous_history['advantages'][sim_step_rand].to(device)
                            new_logprobs, new_values, entropies, _ = model_student(states, memories, actions)                            

                            # Loss
                            ratios = torch.exp(new_logprobs - logprobs)
                            surr1 = ratios * advantages
                            surr2 = torch.clamp(ratios, 1-cf.eps_clip, 1+cf.eps_clip) * advantages
                            loss_actor = -(torch.min(surr1, surr2)).mean()
                            loss_entropy = -cf.entropy * entropies.mean()
                            clipped_values = values + torch.clamp(new_values-values, -cf.value_clip, cf.value_clip)
                            loss_critic = cf.critic * F.mse_loss(clipped_values, returns_targets)
                            loss =  (loss_actor + loss_critic + loss_entropy) / cf.accumulation_steps
                            kl_divergence = ((ratios - 1) - (new_logprobs - logprobs)).mean()
                        _, _, _, memories = model(states, memories) 

                    if perform_update :

                        # Adjust the learning rate
                        dist.all_reduce(kl_divergence, op=dist.ReduceOp.AVG, group=group_agents)
                        lr = tr.update_lr(lr, kl_divergence, kl_threshold=cf.kl_threshold)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

        
                        # Step optimizer
                        scaler.scale(loss).backward()
                        backward_count += 1
                        if backward_count % cf.accumulation_steps == 0 and backward_count != 0:
                            average_gradients(model_student, size-1, group_agents) # ALL REDUCE on scaled loss
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model_student.parameters(), cf.grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                            # Reinitialize
                            backward_count = 0

                    if rank == 1 and perform_update :
                        plotter.add('loss/loss_actor', loss_actor.item(), sim_count*cf.sim_duration+sim_step)
                        plotter.add('loss/loss_critic', loss_critic.item(), sim_count*cf.sim_duration+sim_step)
                        plotter.add('loss/loss_entropy', loss_entropy.item(), sim_count*cf.sim_duration+sim_step)
                        plotter.add('loss/kl_divergence', kl_divergence.item(), sim_count*cf.sim_duration+sim_step)
                        plotter.add('loss/lr', lr, sim_count*cf.sim_duration+sim_step)

                    actions, logprobs, values = [previous_history[key][sim_step].to(device) \
                                                    for key in ['actions', 'logprobs', 'values']]      

            # Gather and process action
            if is_inference:
                actions_star = rearrange(actions_star, 't b d -> b t d').contiguous()
                gather_dummy_src(actions_star, rank, size-1)
                actions_star = rearrange(actions_star, 'b t d -> t b d')
                buffer['actions_star'] += list(torch.unbind( actions_star ))
                if rank == 1:
                    plotter.add('action/mean', (actions_star[0, :, 0]).mean().item(), sim_count*cf.sim_duration+sim_step)
                    plotter.add('action/std', (actions_star[0, :, 0]).std().item(), sim_count*cf.sim_duration+sim_step)
        
                # Act(num_envs, 1)
                if(len(buffer['actions_star'])>0):
                    action = buffer['actions_star'].pop(0)
                    action = torch.clamp(action, min=-1, max=1)
                else:
                    action = torch.dzeros(cf.num_envs, 1)

                # Step environment
                if rank == rank_sim:
                    env.step(action.to(rank_sim))
                    
                if sim_count % 10 == 0 and sim_step % cf.sim_duration == 256 and rank == 0 and cf.render :
                    torchvision.io.write_video('/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/test_'  + str(sim_count) + '_vid_.mp4', 
                                                torch.stack(full_video)[:,1], 8*cf.frames)
            sim_step += 1

        else:
            if rank == rank_sim:
                env.step() #render=cf.render)

    if rank == rank_sim:
        env.close()
        simulation_app.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    cf.path_experiment = tr.setup_run(cf.path_main+'configuration.py', cf.path_experiment)
    
    # logging.basicConfig(filename=cf.path_logs, level=logging.INFO)
    # log_handler = logging.FileHandler(cf.path_logs, mode='a')
    # log_handler.setLevel(logging.INFO)
    # sys.stdout = log_handler.stream
    # sys.stderr = log_handler.stream
    # file_logs = open(cf.path_logs, 'w') 
    # sys.stdout = file_logs

    num_gpus=torch.cuda.device_count()
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["CUDA_ERROR_DUMP"] = "1"
    #os.environ["NCCL_DEBUG_SUBSYS"]="COLL" # print collective calls 
    os.environ["TORCH_CPP_LOG_LEVEL"]="FATAL" # INFO WARNING FATAL
    os.environ["NCCL_DEBUG"]="DEBUG" # WARN INFO DEBUG TRACE
    os.environ["TORCH_DISTRIBUTED_DEBUG"]="INFO"
    os.environ["TORCH_NCCL_ENABLE_MONITORING"]="1"
    
    size = num_gpus
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=isaac_training, args=(cf, rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()