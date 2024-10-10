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
from torch.amp import GradScaler
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append(join(dirname(dirname(os.getcwd()))))
sys.path.append(os.getcwd())

from omni.isaac.lab.app import AppLauncher
from memoryPerceiver.memory_perceiver.main.configuration import cf
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_memories, reset_memories, freeze_parameters
from memoryPerceiver.memory_perceiver.core.distributed import setup_distributed, average_gradients, scatter_dummy_src, gather_dummy_src
from memoryPerceiver.memory_perceiver.core.miscellaneous import custom_random, print_cuda_tensors
import memoryPerceiver.memory_perceiver.isaac.trainer as tr
from memoryPerceiver.memory_perceiver.isaac.scaler import RunningStandardScaler
from memoryPerceiver.memory_perceiver.isaac.plot import Plotter
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCritic, ActorCriticSimple, ActorCriticPerceiver, create_agent
from memoryPerceiver.memory_perceiver.core.rtrl import  RTRL_partial, shuffle_split_indexes

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
        app_launcher = AppLauncher(headless=True, enable_cameras=True)
        simulation_app = app_launcher.app
        import gymnasium as gym
        from omni.isaac.core.utils.torch.maths import set_seed
        import omni.usd
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.stage import open_stage, save_stage
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        enable_extension("omni.isaac.sensor")
        env_cfg = parse_env_cfg(
            cf.task, device=str(device), num_envs=cf.num_envs, use_fabric=False, 
        )
        env = gym.make(cf.task, cfg=env_cfg, render_mode="rgb_array")
        print(env.render_mode)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        seed_isaac = set_seed(cf.seed, torch_deterministic=False)
        env.reset()
        
        value_scaler = RunningStandardScaler(size=1, device=device)  
        if os.path.exists(cf.path_scaler):
            value_scaler.load_state_dict(torch.load(cf.path_scaler, weights_only=True))          

    if rank in rank_agents:
        model = create_agent(cf, rank)
        model_student = create_agent(cf, rank)
        if isinstance(model, ActorCriticPerceiver):
            memories = create_memories(model.actor, cf.num_envs, device=device)
            if not os.path.exists(cf.path_memories) and cf.training_mode != 'RTRL':
                torch.save(memories, cf.path_memories+str(rank)+'.pt')
        else :
            memories = None
            memories_student = None
            # last_memories = torch.zeros_like(memories['cross_cmem'], device=device)
        optimizer = torch.optim.Adam(model_student.parameters(), lr=cf.learning_rate, eps=1e-5)
        lr=cf.learning_rate
        scaler = GradScaler()
        if cf.training_mode == 'Backward':
            freeze_parameters(model_student, 'actor.state_')
        if rank == 1:
            plotter = Plotter('tensorboard', titles=['loss_actor', 'loss_critic', 'loss_entropy', 'actions_gap', 'returns'],
                            path_plots=cf.path_experiment)
            
    if cf.training_mode == 'RTRL':
        if rank == rank_sim:
            slice_indexes_list = shuffle_split_indexes(cf.nb_latents_mem, cf.query_dim, nb_chunks=cf.num_envs)
            len_slice_indexes = torch.tensor(len(slice_indexes_list[0]), device=device)
            slice_indexes = torch.tensor(slice_indexes_list, device=device, dtype=torch.long)
        if rank in rank_agents:
            mem_dim = memories.shape
            model.flattened_mem = mem_dim
            model_student.flattened_mem = mem_dim
            memories = rearrange(memories, 'b l h -> (b l h)')
            if not os.path.exists(cf.path_memories):
                torch.save(memories, cf.path_memories+str(rank)+'.pt')
            len_slice_indexes = torch.tensor(0, device=device)
        dist.broadcast(len_slice_indexes, src=rank_sim)  
        if rank in rank_agents:
            slice_indexes = torch.zeros(cf.num_envs, len_slice_indexes, device=device, dtype=torch.long)
        dist.broadcast(slice_indexes, src=rank_sim)  
        if rank in rank_agents:
            slice_indexes = slice_indexes[rank-1]
            print(f"{rank} {slice_indexes}")
            rtrl_grad = RTRL_partial(model_student, param_state_filter='actor.state_', num_parameters=70000,
                        chunk_size=128, slice_indexes=slice_indexes.tolist())        

    history_keys = ['obs', 'actions', 'values', 'logprobs', 'rewards', 'returns', 'returns_targets', 
                    'advantages', 'birth_time', 'death_index']
    history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, cf.dim_observations, device)
    previous_history = deepcopy(history)
    buffer = {'video':[], 'audio':[], 'obs':[], 'rewards':[], 'actions_star':[]}
    video = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
    next_values = torch.zeros(cf.num_envs, device=rank)
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
                        if isinstance(model, ActorCriticPerceiver):
                            memories = torch.load(cf.path_memories+str(rank)+'.pt', weights_only=True)
                        if sim_count == 0 and os.path.exists(cf.path_model):
                            state_dict = torch.load(cf.path_model, weights_only=True)
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
                    gather_dummy_src(history['values'], rank, size-1, swap_axis=True)
                    print(f"{rank}, {next_values.shape} {next_values.dtype} {next_values}")
                    gather_dummy_src(next_values, rank, size-1)
                    if rank == rank_sim:
                        next_values = value_scaler(next_values, inverse=True) 
                        unscaled_values = value_scaler(history['values'], inverse=True) 
                        print('unscaled_values:', unscaled_values[:, 0])
                        previous_history['returns'] = tr.calculate_returns(history['rewards'], history['death_index'], gamma=cf.gamma)
                        tmp_advantages = tr.calculate_gae(history['rewards'], unscaled_values, next_values, history['death_index'],
                                                          gamma=cf.gamma, gae_lambda=cf.c_lambda)
                        previous_history['values'] = value_scaler(unscaled_values, train=True) # != history['values'] because update
                        previous_history['returns_targets'] = value_scaler(tmp_advantages + unscaled_values, train=True)
                        previous_history['advantages'] = (tmp_advantages - tmp_advantages.mean()) / (tmp_advantages.std() + 1e-8)  #dim=1 keepdim=True
                    for key in ['obs', 'rewards', 'death_index', 'returns', 'values', 'returns_targets', 'advantages']:
                        scatter_dummy_src(previous_history[key], rank, size-1, swap_axis=True)
                    
                    if rank == 1:
                        plotter.add('returns', previous_history['returns'][0, :].mean().item(), sim_count)
                        for key, value in previous_history.items():
                            if key not in ['actions', 'logprobs', 'generator', 'obs']:
                                print(f"{key}: {value[:,0]}")
                            if key == 'actions':
                                print(f"{key}: {value[:,0, 0]}")
                            if key == 'obs':
                                print(f"{key}: {value[:,0, 0]}")
                    if rank in rank_agents and isinstance(model, ActorCriticPerceiver):
                            tmp_memories = memories
                            memories = torch.load(cf.path_memories+str(rank)+'.pt', weights_only=True).detach()
                            memories_student = memories.clone().detach().requires_grad_(True)
                            dist.barrier(group=group_agents)
                            torch.save(tmp_memories, cf.path_memories+str(rank)+'.pt')
                    if rank == rank_sim:
                        print('returns mean', previous_history['returns'].mean())
                        print('values mean', unscaled_values.mean())
                        print('target mean', (tmp_advantages + unscaled_values).mean())
                        print('scaler mean', value_scaler.running_mean)
                        print('scaler var', value_scaler.running_variance)
                        value_scaler.running_variance = torch.clip(value_scaler.running_variance, 0, 200)

                    loss=torch.tensor(0.0, device=device)
                    backward_count= 0
                sim_step = 0

            if is_inference:
                
                ### Get agent observations and dones

                if rank == rank_sim:

                    # obs
                    all_obs = env.unwrapped._get_observations()
                    history['obs'][sim_step] = all_obs['policy'].clone()    # (num_envs, d)
                    if cf.render:
                        video = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                        # ts.save(video[0], '/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/live.jpg')
                        full_video.append(video[8].cpu())
                        video = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
                    
                    #dones
                    dones = env.unwrapped.last_dones.int().clone()
                    if sim_step >= 1:
                        history['birth_time'][sim_step] = history['birth_time'][sim_step-1]
                    history['birth_time'][sim_step, dones==1] = sim_step
                    history['death_index'][sim_step]=dones

                scatter_dummy_src(history['obs'][sim_step], rank, size-1)
                tr.update_buffers(buffer, {'obs': history['obs'][sim_step]}, frames=cf.frames) 
                scatter_dummy_src(history['death_index'][sim_step], rank, size-1) 
                scatter_dummy_src(history['birth_time'][sim_step], rank, size-1) 
                if cf.render:
                    scatter_dummy_src(video, rank, size-1)
                    tr.update_buffers(buffer, {'video': video}, frames=cf.frames)                  

            ### Agent
            
            if sim_step % cf.frames_future == 0 and rank in rank_agents:
                if is_inference:
                    ## Inference, save history and memory update
                    
                    if isinstance(model, ActorCriticPerceiver):
                        dones_list = torch.nonzero(history['death_index'][sim_step].int()).flatten().tolist()
                        reset_memories(memories, indexes=dones_list)
                    obs_stack = torch.stack(buffer['obs'])
                    states=get_states(obs_stack, sim_step, cf, history['birth_time'][sim_step], device)
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        actions, values, logprobs, new_memories = model(states, memories)  # !!!! no update mem 
                        if cf.training_mode != 'Backward':
                            memories = new_memories
                    history['actions'][sim_step] = actions  
                    history['logprobs'][sim_step] = logprobs 
                    history['values'][sim_step:sim_step+cf.frames_future] = values  # for cumulative sum  

                    # Instanciate action before dispatch
                    actions_star = model.instanciate_actions(actions, dt=cf.frames_future).to(torch.float32) # important for gather !!!!

                else:

                    ## Process inputs
                    # sim_step_rand = torch.randint(0, int(1*cf.sim_duration), (1,)).item()   
                    sim_step_rand=sim_step 
                    actions, logprobs, values, birth_time, death_index = [previous_history[key][sim_step_rand].to(device) \
                                                for key in ['actions', 'logprobs', 'values', 'birth_time', 'death_index']]
                    if isinstance(model, ActorCriticPerceiver):
                        dones_list = torch.nonzero(death_index.int()).flatten().tolist()
                        reset_memories(memories, indexes=dones_list) 
                    obs_stack = previous_history['obs'][sim_step_rand-cf.frames+1:sim_step_rand+1]
                    states=get_states(obs_stack, sim_step_rand, cf, birth_time, device)

                    perform_update = ( custom_random(sim_count, sim_step_rand) < cf.update_probability and \
                                        sim_step_rand <= cf.sim_duration-1 ) or cf.training_mode == 'TBPTT'
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        if perform_update:
                            returns_targets = previous_history['returns_targets'][sim_step_rand].to(device)
                            advantages = previous_history['advantages'][sim_step_rand].to(device)
                            if cf.training_mode in ['Backward', 'RTRL'] and isinstance(model, ActorCriticPerceiver):
                                memories_student = memories.clone().detach().requires_grad_(True)
                            new_logprobs, new_values, entropies, new_memories_student = model_student(states, memories_student, actions)     
                            if cf.training_mode == 'TBPTT':
                                memories_student = new_memories_student

                            # Loss
                            ratios = torch.exp(new_logprobs - logprobs)
                            surr1 = ratios * advantages
                            surr2 = torch.clamp(ratios, 1-cf.eps_clip, 1+cf.eps_clip) * advantages
                            loss_actor = -(torch.min(surr1, surr2)).mean()
                            loss_entropy = -cf.entropy * entropies.mean()
                            clipped_values = values + torch.clamp(new_values-values, -cf.value_clip, cf.value_clip)
                            loss_critic = cf.critic * F.mse_loss(clipped_values, returns_targets)
                            new_loss = (loss_actor + loss_critic + loss_entropy) / cf.accumulation_steps
                            kl_divergence = ((ratios - 1) - (new_logprobs - logprobs)).mean()
                            if kl_divergence > 3*cf.kl_threshold:
                                new_loss *= 0
                            loss += new_loss
                        if cf.training_mode != 'Backward':
                            _, _, _, memories = model(states, memories) 

                    if perform_update :

                        # Adjust the learning rate
                        dist.all_reduce(kl_divergence, op=dist.ReduceOp.SUM, group=group_agents)
                        if cf.scheduler == 'kl_div':
                            lr = tr.lr_kl_scheduler(lr, kl_divergence, kl_threshold=cf.kl_threshold, min_lr=cf.range_lr[0], max_lr=cf.range_lr[1])
                        if cf.scheduler == 'warm-up':
                            lr = tr.lr_warm_up(2*sim_count, min_lr=cf.range_lr[0], max_lr=cf.range_lr[1])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # Step optimizer
                        if cf.training_mode == 'Backward':
                            scaler.scale(loss).backward()
                            loss = torch.tensor(0.0, device=device)
                        if cf.training_mode == 'RTRL':
                            loss = scaler.scale(loss)
                            rtrl_grad.step(model_student, states, memories_student, loss, old_actions=actions)
                            loss = torch.tensor(0.0, device=device)
                        backward_count += 1
                        if backward_count % cf.accumulation_steps == 0 and backward_count != 0:
                            if cf.training_mode == 'TBPTT':
                                scaler.scale(loss).backward()
                                loss = torch.tensor(0.0, device=device)
                                if isinstance(model, ActorCriticPerceiver):
                                    memories_student = memories.clone().detach().requires_grad_(True)
                            if cf.training_mode == 'RTRL':
                                average_gradients(model_student, size-1, group_agents, param_filter='actor.out_')
                            else:
                                average_gradients(model_student, size-1, group_agents)  # ALL REDUCE on scaled loss
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

            # Gather action and step env
            if is_inference:
                gather_dummy_src(actions_star, rank, size-1, swap_axis=True)
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
                    history['rewards'][sim_step] = env.unwrapped._get_rewards()    # (num_envs, 1)

                # Get next values at the last iteration
                
                if sim_step == cf.sim_duration-1:
                    next_obs = torch.zeros(cf.num_envs, cf.dim_observations, device=rank)
                    next_birth_time = torch.zeros(cf.num_envs, device=rank)
                    if rank == rank_sim:
                        all_obs = env.unwrapped._get_observations()
                        next_obs = all_obs['policy'].clone()    # (num_envs, d)
                        if cf.render:
                            video = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                            video = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
                            
                        dones = env.unwrapped.last_dones.int().clone()
                        next_birth_time = history['birth_time'][sim_step-1]
                        history['birth_time'][sim_step, dones==1] = sim_step
                        history['death_index'][sim_step]=dones
                    scatter_dummy_src(next_obs, rank, size-1) 
                    scatter_dummy_src(next_birth_time, rank, size-1) 
                    tr.update_buffers(buffer, {'obs': next_obs}, frames=cf.frames)
                    if cf.render:
                        scatter_dummy_src(video, rank, size-1)
                        tr.update_buffers(buffer, {'video': video}, frames=cf.frames)
                    if rank in rank_agents:
                        obs_stack = torch.stack(buffer['obs'])
                        states=get_states(obs_stack, sim_step, cf, next_birth_time, device)
                        with autocast(device_type='cuda', dtype=torch.bfloat16):
                            _, next_values, _, _ = model_student(states, memories) 
                        next_values = next_values.to(torch.float32)
                        
                if sim_count % 50 == 0 and sim_step == cf.sim_duration-1 and rank == 0 and cf.render :
                    torchvision.io.write_video('/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/test_'  + str(sim_count) + '_vid_.mp4', 
                                                torch.stack(full_video), 8*cf.frames)
            sim_step += 1

        else:
            if rank == rank_sim:
                env.step()

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