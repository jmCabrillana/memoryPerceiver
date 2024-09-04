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

# from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
# from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
# from omniisaacgymenvs.utils.task_util import initialize_task
# from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omni.isaac.lab.app import AppLauncher

from memoryPerceiver.memory_perceiver.main.configuration import cf
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_memories, reset_memories
from memoryPerceiver.memory_perceiver.core.distributed import setup_distributed, average_gradients, scatter_dummy_src, gather_dummy_src
from memoryPerceiver.memory_perceiver.core.miscellaneous import custom_random, print_cuda_tensors
import memoryPerceiver.memory_perceiver.isaac.trainer as tr
from memoryPerceiver.memory_perceiver.isaac.plot import Plotter
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCritic, ActorCriticSimple, ActorCriticPerceiver, create_agent


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
    if rank in rank_agents:
        model = create_agent(cf, rank)
        model_student = create_agent(cf, rank)
        if isinstance(model, ActorCriticPerceiver):
            memories = create_memories(model.actor, cf.num_envs, device=device, 
                                       mem_size=cf.mem_size, cmem_size=cf.cmem_size)
        else :
            memories = None
        # last_memories = torch.zeros_like(memories['cross_cmem'], device=device)

    history_keys = ['actions', 'values', 'logprobs', 'returns', 'returns_targets', 'rewards', 'advantages', 'death_index', 'generator']
    history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, device)
    previous_history = deepcopy(history)
    buffer = {'video':[], 'audio':[], 'obs':[], 'rewards':[], 'actions_star':[]}
    obs = torch.zeros(cf.num_envs, cf.dim_observations, device=rank)
    video = torch.zeros(cf.num_envs, cf.height, cf.width, cf.channels, device=rank, dtype=torch.uint8)
    reward = torch.zeros(cf.num_envs, 1, device=rank)
    actions_star = torch.zeros(cf.frames_future, cf.num_envs, cf.dim_actions, device=rank)

    if rank == rank_sim:
        seed_isaac = set_seed(cf.seed, torch_deterministic=False)
        env.reset()
    if rank in rank_agents:
        optimizer = torch.optim.Adam(model_student.parameters(), lr=cf.learning_rate)
        if os.path.exists(cf.path_model):
            optimizer.load_state_dict(torch.load(cf.path_optimizer))
        scaler = GradScaler()
        if rank == 1:
            plotter = Plotter('tensorboard', titles=['loss_actor', 'loss_critic', 'loss_entropy', 'actions_gap', 'returns'],
                                path_plots=cf.path_experiment)
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
                if rank == rank_sim:
                    env.unwrapped.sim.reset(soft=True) #False
                
                if rank == rank_sim:
                    if len(history['rewards'])>0:
                        print(f"$$$ r{rank} CUMUL REWARD: {history['rewards'].sum().item()} $$$")
                    print(f"SEED: {seed}")
                    print(f"SIM_COUNT: {sim_count}")
                    if is_inference:
                        print('PLAY')
                    else:
                        print('REPLAY')

                if is_inference:
                    if rank in rank_agents:
                        # print(f"MEM:  {F.mse_loss(last_memories, memories['cross_cmem']).item()}")
                        if sim_count == 0 and os.path.exists(cf.path_model):
                            model.load_state_dict(torch.load(cf.path_model)) #, map_location=device))
                            model_student.load_state_dict(torch.load(cf.path_model)) #, map_location=device))
                        else:
                            if sim_count % ( (cf.num_replay+1)*cf.student_update_step ) == 0 :
                                model.load_state_dict(model_student.state_dict())
                        dist.barrier(group=group_agents)
                        if rank == 1:
                            torch.save(model_student.state_dict(), cf.path_model)
                            torch.save(optimizer.state_dict(), cf.path_optimizer)
                        model.eval()
                        history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, device)
                else:
                    for key in ['rewards', 'death_index']:
                        swap_batch_time_tmp = rearrange(history[key], 't b -> b t').contiguous()
                        scatter_dummy_src(swap_batch_time_tmp, rank, size-1)
                        history[key] = rearrange(swap_batch_time_tmp, 'b t -> t b')
                    if rank in rank_agents:
                        # last_memories = memories['cross_cmem'].clone()
                        model_student.train() 

                    previous_history = deepcopy(history)
                    previous_history['returns'] = tr.calculate_returns(history['rewards'], history['death_index'], gamma=cf.gamma)
                    tmp_advantages = tr.calculate_gae(history['rewards'], history['values'], 
                                                        history['death_index'], gamma=cf.gamma, gae_lambda=cf.c_lambda)
                    # previous_history['advantages'] = (tmp_advantages - tmp_advantages.mean(dim=1, keepdim=True)) / tmp_advantages.std(dim=1, keepdim=True)
                    previous_history['advantages'] = tmp_advantages
                    previous_history['returns_targets'] = tmp_advantages + history['values']
                    # previous_history['advantages'] = previous_history['returns'] - previous_history['values']
                    
                    if rank == 1:
                        plotter.add('returns', previous_history['returns'][0, :].mean().item(), sim_count)
                        for key, value in previous_history.items():
                            if key not in ['actions', 'logprobs', 'returns', 'generator']:
                                print(f"{key}: {value[:,0]}")
                            if key == 'actions':
                                print(f"{key}: {value[:,0, 0]}")
                                #print(f"{key}1: {value[:,0, 1]}")
                            if key == 'returns':
                                print(f"{key}: {(value[:,0]*10).int()}")
                        
                if rank in rank_agents:
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    if isinstance(model, ActorCriticPerceiver):
                        reset_memories(memories)
                tr.set_seeds(seed)
                full_video = []
                birth_time = torch.zeros(cf.num_envs, device=rank)
                for key in buffer.keys():
                    buffer[key]=[]
                backward_count= 0
                sim_step = 0

            ### Get agent observations 

            if rank == rank_sim:
                all_obs = env.unwrapped._get_observations()
                obs = all_obs['policy'].clone()    # (num_envs, d)
                if cf.render:
                    video = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                    #ts.save(video[0], '/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/live.jpg')
                    full_video.append(video.cpu())
                reward=env.unwrapped._get_rewards()    # (num_envs, 1)
                if is_inference:
                    history['rewards'][sim_step] = reward
            for x in [obs, reward]: #video
                scatter_dummy_src(x, rank, size-1) 
            tr.update_buffers(buffer, {'obs': obs, 'video': video, 'rewards': reward}, frames=cf.frames)
            if cf.render:
                scatter_dummy_src(video, rank, size-1)
                tr.update_buffers(buffer, {'video': video}, frames=cf.frames)

            ### Reset memory at death and record 

            dones = torch.zeros(cf.num_envs, device=rank)
            if rank == rank_sim:
                dones, _ = env.unwrapped._get_dones()
                dones = dones.int().clone()
                birth_time[dones==1] = sim_step

                # Update death dictionnary
                if is_inference:
                    history['death_index'][sim_step, :]=dones
            scatter_dummy_src(dones, rank, size-1) 
            scatter_dummy_src(birth_time, rank, size-1) 
            dones_list = torch.nonzero(dones.int()).flatten().tolist()
            if rank in rank_agents and isinstance(model, ActorCriticPerceiver):
                reset_memories(memories, indexes=dones_list)

            ### Agent
            
            if len(buffer['actions_star'])==0 :
                if rank in rank_agents:

                    ## Process inputs

                    states={}
                    mask = tr.create_mask(birth_time, sim_step, frames=cf.frames)
                    states['observations'] = tr.buffer_padding(torch.stack(buffer['obs']), [cf.dim_observations],
                                                                num_envs=cf.num_envs, frames=cf.frames)
                    states['observations'][mask] *= 0
                    if cf.input_video:
                        states['video'] = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
                        # states['video'] = tr.buffer_padding(torch.stack(buffer['video']), [cf.height, cf.width, 3], num_envs=cf.num_envs, frames=cf.frames)
                        states['video'][mask] *= 0
                        states['video'] = rearrange(states['video'], 'b t h w c -> b c t h w')
                    if cf.input_audio:
                        states['audio'] = torch.zeros(cf.num_envs, cf.len_audio_raw, device=device)

                    ## Inference, save history and memory update

                    if is_inference:
                        history['generator'][sim_step, rank] = torch.cuda.get_rng_state(rank).clone() 
                        with autocast(device_type='cuda', dtype=torch.bfloat16):
                            actions, logprobs, values, memories = model(states, memories)   
                        history['actions'][sim_step] = actions  
                        history['logprobs'][sim_step] = logprobs 
                        history['values'][sim_step:sim_step+cf.frames_future] = values  # for cumulative sum  
                    else:
                        actions, logprobs, values = [previous_history[key][sim_step].to(device) \
                                                    for key in ['actions', 'logprobs', 'values']]      

                    ## Backward pass

                    if not is_inference :

                        torch.cuda.set_rng_state(previous_history['generator'][sim_step, rank], device=device)
                        with autocast(device_type='cuda', dtype=torch.bfloat16):
                            if custom_random(sim_count, sim_step) < cf.update_probability :
                                returns = previous_history['returns_targets'][sim_step].to(device)
                                advantages = previous_history['advantages'][sim_step].to(device)
                                new_logprobs, new_values, entropies, _ = model_student(states, memories, actions)                            

                                # Loss
                                ratios = torch.exp(new_logprobs - logprobs)
                                surr1 = ratios * advantages
                                surr2 = torch.clamp(ratios, 1-cf.eps_clip, 1+cf.eps_clip) * advantages
                                loss_actor = -(torch.min(surr1, surr2)).mean()
                                loss_entropy = -cf.entropy * entropies.mean()
                                loss_critic = cf.critic * F.mse_loss(new_values, returns)
                                loss =  (loss_actor + loss_critic + loss_entropy) / cf.accumulation_steps

                            _, _, _, memories = model(states, memories) 

                        if custom_random(sim_count, sim_step) < cf.update_probability :
                            scaler.scale(loss).backward()
                            backward_count += 1


                            # Step optimizer

                            if backward_count % cf.accumulation_steps == 0 and backward_count != 0:
                                average_gradients(model_student, size-1, group_agents) # ALL REDUCE on scaled loss
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model_student.parameters(), cf.grad_norm)
                                scaler.step(optimizer)
                                # for name, param in model_student.named_parameters():
                                #     if True or torch.isnan(param).any():
                                #         print(f'{name} {param.data}')
                                #         print(f'grad {param.grad.data}')
                                scaler.update()
                                optimizer.zero_grad()

                                # Reinitialize
                                backward_count = 0

                        if rank == 1 and custom_random(sim_count, sim_step) < cf.update_probability :
                            plotter.add('loss_actor', loss_actor.item(), sim_count*cf.sim_duration+sim_step)
                            plotter.add('loss_critic', loss_critic.item(), sim_count*cf.sim_duration+sim_step)
                            plotter.add('loss_entropy', loss_entropy.item(), sim_count*cf.sim_duration+sim_step)
                    # Update buffers

                    actions_star = model.instanciate_actions(actions, dt=cf.frames_future).to(torch.float32) # important for gather !!!!
                    
                actions_star = rearrange(actions_star, 't b d -> b t d').contiguous()
                gather_dummy_src(actions_star, rank, size-1)
                actions_star = rearrange(actions_star, 'b t d -> t b d')
                buffer['actions_star'] += list(torch.unbind( actions_star ))
                if rank == 1:
                    actions_gap =  (actions_star[0, :, 0]).mean() # - actions_star[1, :, 0]).mean()
                    plotter.add('actions_gap', actions_gap.item(), sim_count*cf.sim_duration+sim_step)
            
            # Act(num_envs, 1)
            if(len(buffer['actions_star'])>0):
                action = buffer['actions_star'].pop(0)
                action = torch.clamp(action, min=-1, max=1)
            else:
                action = torch.dzeros(cf.num_envs, 1)

            if rank == rank_sim:

                # Step environment
                if sim_step == 0:
                    pass #env.unwrapped.__init__(env_cfg)
                env.step(action.to(rank_sim))
                # env.step(render=cf.render)
                # env.sim_frame_count += 1
                # env._task.post_physics_step()
            dist.barrier()

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