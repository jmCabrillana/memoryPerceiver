from os.path import join, dirname
import os
import sys
import gc
import logging
import argparse
from copy import deepcopy
import numpy as np
import time
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

from isaaclab.app import AppLauncher
from memoryPerceiver.memory_perceiver.main.configuration import isaac_cfg
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_memories, reset_memories, freeze_parameters
from memoryPerceiver.memory_perceiver.core.distributed import setup_distributed, average_gradients, scatter_dummy_src, gather_dummy_src
from memoryPerceiver.memory_perceiver.core.miscellaneous import custom_random, print_cuda_tensors
import memoryPerceiver.memory_perceiver.isaac.trainer as tr
from memoryPerceiver.memory_perceiver.isaac.scaler import RunningStandardScaler
from memoryPerceiver.memory_perceiver.isaac.plot import Plotter
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCriticSimple, ActorCriticPerceiver, create_agent
from memoryPerceiver.memory_perceiver.core.rtrl import  RTRL_partial, shuffle_split_indexes


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
        app_launcher = AppLauncher(headless=False, enable_cameras=True)
        simulation_app = app_launcher.app
        import gymnasium as gym
        from isaacsim.core.utils.torch.maths import set_seed
        # import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        # from isaac.core.utils.stage import open_stage, save_stage
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        enable_extension("omni.isaac.sensor")
        env_cfg = parse_env_cfg(
            cf.task, device=str(device), num_envs=cf.num_envs, use_fabric=False, 
        )
        env = gym.make(cf.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        seed_isaac = set_seed(cf.seed, torch_deterministic=False)
        env.reset()
        value_scaler = RunningStandardScaler(size=1, device=device)  
        if os.path.exists(cf.path_scaler):
            value_scaler.load_state_dict(torch.load(cf.path_scaler, weights_only=True))   

    if rank in rank_agents:
        model = create_agent(cf.agent_type, cf, rank)
        model_student = create_agent(cf.agent_type, cf, rank)
        optimizer = torch.optim.Adam(model_student.parameters(), lr=cf.learning_rate, eps=1e-5)
        if os.path.exists(cf.path_model):
            state_dict = torch.load(cf.path_model, weights_only=True)
            model.load_state_dict(state_dict['model'])
            model_student.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
        if isinstance(model, ActorCriticPerceiver):
            memories = create_memories(model.actor, cf.num_envs, device=device)
            path_mem = str(cf.path_memories)+str(rank)+'.pt'
            if os.path.exists(path_mem):
                memories = torch.load(path_mem, weights_only=True)
            memories_student = memories.clone().detach().requires_grad_(True)
            mem_dim = None
        else :
            memories = None
            memories_student = None
        if cf.tutor_type is not None:
            model_tutor = create_agent(cf.tutor_type, cf, rank)
            state_dict = torch.load(cf.path_model_tutor, weights_only=True)
            model_tutor.load_state_dict(state_dict['model'])
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
            memories_student = memories.clone().detach().requires_grad_(True)
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

    history_keys = ['obs', 'actions', 'actions_mean', 'actions_std', 'actions_star', 'values', 'logprobs', 'rewards', 'returns', 'returns_targets', 
                    'advantages', 'birth_time', 'death_index']
    history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, cf.dim_observations, device)
    buffer = {'obs':[], 'video':[], 'audio':[], 'birth_time': [], 
              'actions_star':[torch.zeros(cf.num_envs, cf.dim_actions, device=device)], }
    path_buffer_old = str(cf.path_buffer)+'_old'+str(rank)+'.pt'
    path_buffer_new = str(cf.path_buffer)+'_new'+str(rank)+'.pt'
    if not os.path.exists(path_buffer_new):
        torch.save(buffer, path_buffer_new)
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
                
                if rank == rank_sim:
                    if len(history['rewards'])>0:
                        print(f"$$$ r{rank} CUMUL REWARD: {history['rewards'].sum().item()} $$$")
                    print(f"SEED: {seed}")
                    print(f"SIM_COUNT: {sim_count}")

                if is_inference:
                    if rank in rank_agents:
                        if sim_count % ( (cf.num_replay+1)*cf.student_update_step ) == 0 :
                            model.load_state_dict(model_student.state_dict())
                            model.eval()
                        if rank in rank_agents and isinstance(model, ActorCriticPerceiver):
                            if cf.training_mode == 'RTRL':
                                b, l, h = mem_dim
                                memories_save = rearrange(memories, '(b l h) -> b l h', b=b, l=l, h=h)
                                torch.save(memories_save, path_mem)
                            else:
                                torch.save(memories, path_mem)
                        dist.barrier(group=group_agents)
                        if rank == 1:
                            torch.save({'model': model_student.state_dict(), 'optimizer': optimizer.state_dict()}, cf.path_model)
                        history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, cf.dim_observations, device)
                    if rank == rank_sim:
                        torch.save(value_scaler.state_dict(), cf.path_scaler)
                    full_video = []
                    kl_list = []
                    buffer = torch.load(path_buffer_new, weights_only=True)
                    torch.save(buffer, path_buffer_old) # end of previous inference
                else:
                    if sim_count % (cf.num_replay+1) == 1:
                        torch.save(buffer, path_buffer_new) # end of new inference

                        # Compute generalized advantage

                        gather_dummy_src(history['values'], rank, size-1, swap_axis=True)
                        gather_dummy_src(next_values, rank, size-1)
                        if rank == rank_sim:
                            unscaled_next_values = value_scaler(next_values, inverse=True) 
                            unscaled_values = value_scaler(history['values'], inverse=True) 
                            history['returns'] = tr.calculate_returns(history['rewards'], history['death_index'], gamma=cf.gamma)
                            tmp_advantages = tr.calculate_gae(history['rewards'], unscaled_values, unscaled_next_values, history['death_index'],
                                                            gamma=cf.gamma, gae_lambda=cf.c_lambda)
                            history['values'] = value_scaler(unscaled_values, train=True) # != history['values'] because update
                            history['returns_targets'] = value_scaler(tmp_advantages + unscaled_values, train=True)
                            history['advantages'] = (tmp_advantages - tmp_advantages.mean()) / (tmp_advantages.std() + 1e-8)  #dim=1 keepdim=True
                        for key in ['returns', 'values', 'returns_targets', 'advantages']:
                            scatter_dummy_src(history[key], rank, size-1, swap_axis=True)
                        # if rank == rank_sim:
                        #     print('returns mean', history['returns'].mean())
                        #     print('values mean', unscaled_values.mean())
                        #     print('target mean', (tmp_advantages + unscaled_values).mean())
                        #     print('scaler mean', value_scaler.running_mean)
                        #     print('scaler var', value_scaler.running_variance)
                        #     print('unscaled_values:', unscaled_values[:, 0])
                        

                    dist.barrier()
                    if rank in rank_agents and isinstance(model, ActorCriticPerceiver):
                        memories = torch.load(path_mem, weights_only=True).detach()
                        if cf.training_mode == 'RTRL':
                            memories = rearrange(memories, 'b l h -> (b l h)')
                        memories_student = memories.clone().detach().requires_grad_(True)

                    if rank == 1:
                        plotter.add('returns', history['returns'][0, :].mean().item(), sim_count)
                        plotter.add('action/mean', history['actions'][:, :, 0].mean().item(), sim_count)
                        plotter.add('action/std', history['actions'][:, :, 0].std().item(), sim_count)
                        plotter.add('loss/lr', lr, sim_count)
                    #     for key, value in history.items():
                    #         if key not in ['actions', 'actions_star', 'logprobs', 'generator', 'obs', 'rewards']:
                    #             print(f"{key}: {value[:,0]}")
                    #         if key == 'actions':
                    #             print(f"{key}: {value[:,0, 0]}")
                    #         if key == 'actions_star':
                    #             print(f"{key}: {value[:,0, 0]}")
                    #         if key == 'obs':
                    #             print(f"{key}: {value[:,0, 0]}")

                    buffer = torch.load(path_buffer_old, weights_only=True)
                    loss_cumul=torch.tensor(0.0, device=device)
                    backward_count=0
                    actor_losses, critic_losses, entropy_losses, kl_tutor_losses = [], [], [], []
                sim_step = 0
            
            sim_count_total = sim_count*cf.sim_duration+sim_step
            if is_inference:
                
                ### Get agent observations and dones

                if rank == rank_sim:

                    # obs
                    all_obs = env.unwrapped._get_observations()
                    history['obs'][sim_step] = all_obs['policy'].clone()    # (num_envs, d)
                    if 'video' in cf.inputs:
                        video = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                        # ts.save(video[0], '/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/live.jpg')
                        # video = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
                    if cf.render:
                        full_video.append(env.render().cpu())
                    
                    # dones
                    dones = env.unwrapped.last_dones.int().clone()
                    if sim_count_total == 0:
                        history['birth_time'][sim_step] = torch.zeros(cf.num_envs, device=device)
                    else:
                        history['birth_time'][sim_step] = buffer['birth_time'][-1].clone()
                    history['birth_time'][sim_step, dones==1] = sim_count_total
                    history['death_index'][sim_step]=dones

                # Scatter for agents
                scatter_dummy_src(history['death_index'][sim_step], rank, size-1)
                history['actions_star'][sim_step] = buffer['actions_star'][-1].clone()
                last_state = {}
                for key in ['birth_time'] + cf.inputs:
                    scatter_dummy_src(history[key][sim_step], rank, size-1)
                    last_state[key] = history[key][sim_step]
                tr.buffers_update(buffer, last_state, frames=cf.frames)                 


            ### Agents
            
            if sim_step % cf.frames_future == 0 and rank in rank_agents:

                ## Act
                if is_inference:
                    ## Inference, save history and memory update
                    
                    if isinstance(model, ActorCriticPerceiver):
                        dones_list = torch.nonzero(history['death_index'][sim_step].int()).flatten().tolist()
                        reset_memories(memories, indexes=dones_list, mem_dim=mem_dim)
                    states=tr.mask_states(tr.stack_buffer(buffer), sim_count_total, cf, history['birth_time'][sim_step], device)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        model_outputs = model(states, memories)
                        actions, logprobs, values, new_memories = [model_outputs[key] for key in ['actions', 'logprobs', 'values', 'memories']]
                        if cf.training_mode != 'Backward':
                            memories = new_memories
                        if cf.tutor_type is not None:
                            model_tutor_outputs = model_tutor(states, None)
                            history['actions_mean'][sim_step] = model_tutor_outputs['actions_mean'].detach()
                            history['actions_std'][sim_step] = model_tutor_outputs['actions_std'].detach()
                    history['actions'][sim_step] = actions.detach()
                    history['logprobs'][sim_step] = logprobs.detach()
                    history['values'][sim_step:sim_step+cf.frames_future] = values.detach()  # for cumulative sum  

                    # Instanciate action before dispatch
                    actions_star = model.instanciate_actions(actions.detach(), dt=cf.frames_future).to(torch.float32) # important for gather

                ## Train
                else:
                    
                    # sim_step_rand = torch.randint(0, int(1*cf.sim_duration), (1,)).item()  
                    # sim_count_total = sim_count*cf.sim_duration + sim_step
                    sim_step_rand=sim_step 
                    actions, actions_mean, actions_std, logprobs, values, birth_time, death_index = [history[key][sim_step_rand].to(device) \
                        for key in ['actions', 'actions_mean', 'actions_std', 'logprobs', 'values', 'birth_time', 'death_index']]
                    if isinstance(model, ActorCriticPerceiver):
                        dones_list = torch.nonzero(death_index.int()).flatten().tolist()
                        reset_memories(memories, indexes=dones_list, mem_dim=mem_dim)
                    for key in ['birth_time'] + cf.inputs:
                        last_state[key] = history[key][sim_step]
                    tr.buffers_update(buffer, last_state, frames=cf.frames)  
                    states=tr.mask_states(tr.stack_buffer(buffer), sim_count_total, cf, birth_time, device)

                    perform_update = ( custom_random(sim_count, sim_step_rand) <= cf.update_probability and \
                                        sim_step_rand <= cf.sim_duration-1 ) or cf.training_mode == 'TBPTT'
                    perform_update = perform_update and seed >= cf.start_step
                    with autocast(device_type='cuda', dtype=torch.float16):
                        if perform_update:
                            returns_targets = history['returns_targets'][sim_step_rand].to(device)
                            advantages = history['advantages'][sim_step_rand].to(device)
                            if cf.training_mode in ['Backward', 'RTRL'] and isinstance(model, ActorCriticPerceiver):
                                memories_student = memories.clone().detach().requires_grad_(True)
                            model_student_outputs = model_student(states, memories_student, actions)
                            new_actions_mean, new_actions_std, new_logprobs, new_values, entropies, new_memories_student = \
                                [model_student_outputs[key] for key in ['actions_mean', 'actions_std', 'logprobs', 'values', 'entropies', 'memories']]
                            if cf.training_mode == 'TBPTT':
                                memories_student = new_memories_student

                            # Loss
                            ratios = torch.exp(new_logprobs - logprobs)
                            surr1 = ratios * advantages
                            surr2 = torch.clamp(ratios, 1-cf.eps_clip, 1+cf.eps_clip) * advantages
                            loss_actor = -(torch.min(surr1, surr2)).mean()
                            loss_entropy = -cf.c_entropy * entropies.mean()
                            clipped_values = values + torch.clamp(new_values-values, -cf.value_clip, cf.value_clip)
                            loss_critic = cf.c_critic * F.mse_loss(clipped_values, returns_targets)
                            loss_kl_tutor = torch.tensor(0.0, device=device)
                            if cf.tutor_type is not None:
                                # loss_kl_tutor = F.mse_loss(new_actions_mean, actions_mean)
                                mu1, mu2, sig1, sig2 = new_actions_mean, actions_mean, new_actions_std, actions_std
                                loss_kl_tutor = cf.c_kl_tutor*0.5*( ( (mu2-mu1)/sig2 )**2 + (sig1/sig2)**2 - 2*torch.log(sig1) + 2*torch.log(sig2) - 1 ).mean(dim=1).mean()
                            loss = (loss_actor + loss_critic + loss_entropy + loss_kl_tutor) / cf.accumulation_steps
                            loss_cumul += loss
                            with torch.no_grad():
                                kl_divergence = ((ratios - 1) - (new_logprobs - logprobs)).mean()
                                kl_list.append(kl_divergence.item())
                            if kl_divergence > 3*cf.kl_threshold:
                                loss *= 0
                                loss_cumul *= 0
                        if cf.training_mode != 'Backward':
                            memories = model(states, memories)['memories']

                    if perform_update :

                        # Step optimizer
                        if cf.training_mode == 'Backward':
                            scaler.scale(loss).backward()
                        if cf.training_mode == 'RTRL':
                            loss = scaler.scale(loss)
                            rtrl_grad.step(model_student, states, memories_student, loss, old_actions=actions)
                        backward_count += 1
                        if backward_count % cf.accumulation_steps == 0 and backward_count != 0:
                            if cf.training_mode == 'TBPTT':
                                scaler.scale(loss_cumul).backward()
                                loss_cumul = torch.tensor(0.0, device=device, requires_grad=True)
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
                        actor_losses.append(loss_actor.item())
                        critic_losses.append(loss_critic.item())
                        entropy_losses.append(loss_entropy.item())
                        kl_tutor_losses.append(loss_kl_tutor.item())

            # Gather action and step env
            if is_inference:

                # Step environment with action
                gather_dummy_src(actions_star, rank, size-1, swap_axis=True)
                tr.buffer_update(buffer, 'actions_star', list(torch.unbind( actions_star.detach() )), frames=cf.frames)
                if rank == rank_sim:
                    env.step(buffer['actions_star'][-1].clone())
                    history['rewards'][sim_step] = cf.reward_scale*env.unwrapped._get_rewards()

                # Get next values at the last iteration
                
                if sim_step == cf.sim_duration-1:
                    next_sim_step = sim_step+1
                    next_state = {}
                    next_state['obs'] = torch.zeros(cf.num_envs, cf.dim_observations, device=rank)
                    next_state['birth_time'] = torch.zeros(cf.num_envs, device=rank)
                    if 'actions_star' in cf.inputs:
                        next_state['actions_star'] = buffer['actions_star'][-1].clone()
                    if 'video' in cf.inputs:
                        next_state['video'] = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
                    if 'audio' in cf.inputs:
                        next_state['audio'] = torch.zeros(cf.num_envs, cf.len_audio, device=device)
                    
                    if rank == rank_sim:
                        all_obs = env.unwrapped._get_observations()
                        next_state['obs'] = all_obs['policy'].clone()    # (num_envs, d)
                        if 'video' in cf.inputs:
                            next_state['video'] = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                        dones = env.unwrapped.last_dones.int().clone()
                        next_state['birth_time'] = buffer['birth_time'][-1].clone()
                        next_state['birth_time'][dones==1] = sim_count*cf.sim_duration+next_sim_step
                    last_state = {}
                    for key in ['birth_time'] + cf.inputs:
                        scatter_dummy_src(next_state[key], rank, size-1)
                        last_state[key] = next_state[key]
                    tr.buffers_update(buffer, last_state, frames=cf.frames)
                    if rank in rank_agents:
                        states=tr.mask_states(tr.stack_buffer(buffer), sim_count*cf.sim_duration+next_sim_step, cf, next_state['birth_time'], device)
                        with autocast(device_type='cuda', dtype=torch.float16):
                            next_values = model(states, memories)['values']
                        next_values = next_values.to(torch.float32)

                # video
                if sim_count % 100 == 0 and sim_step == cf.sim_duration-1 and rank == 0 and cf.render :
                    torchvision.io.write_video( cf.path_monitor / ('test_'  + str(sim_count) + '_vid_.mp4'), 
                                                torch.stack(full_video), 8*cf.frames)
            else:

                # Adjust the learning rate after each training

                if sim_step == cf.sim_duration-1 and rank in rank_agents and seed >= cf.start_step :
                    
                    kl = torch.tensor(kl_list, device=device).mean()
                    dist.all_reduce(kl, op=dist.ReduceOp.SUM, group=group_agents)
                    kl /= (size-1)
                    if rank == 1:
                        plotter.add('loss/kl_divergence', kl.item(), sim_count)
                    if cf.scheduler == 'kl_div':
                        lr = tr.lr_kl_scheduler(lr, kl, kl_threshold=cf.kl_threshold, min_lr=cf.range_lr[0], max_lr=cf.range_lr[1])
                    if cf.scheduler.startswith('warm-up'):
                        lr = tr.lr_warm_up(seed, min_lr=cf.range_lr[0], max_lr=cf.range_lr[1], 
                                           steps_warm_up=int(cf.scheduler.split('_')[-1]))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                # Plot losses

                if rank == 1 and sim_step == cf.sim_duration-1:
                    plotter.add('loss/lr', lr, sim_count)
                    plotter.add('loss/actor', np.mean(actor_losses), sim_count)
                    plotter.add('loss/critic', np.mean(critic_losses), sim_count)
                    plotter.add('loss/entropy', np.mean(entropy_losses), sim_count)
                    plotter.add('loss/kl_tutor', np.mean(kl_tutor_losses), sim_count)
            sim_step += 1

        else:
            if rank == rank_sim:
                time.sleep(0.01)
                # env.step()

    if rank == rank_sim:
        env.close()
        simulation_app.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    isaac_cfg.path_experiment = tr.setup_run(isaac_cfg.path_main / 'configuration.py', isaac_cfg.path_experiment)
    os.makedirs(isaac_cfg.path_experiment)
    os.makedirs(isaac_cfg.path_weights, exist_ok=True)
    
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
    os.environ["NCCL_DEBUG"]="WARN" # WARN INFO DEBUG TRACE
    os.environ["TORCH_DISTRIBUTED_DEBUG"]="OFF" # OFF INFO DETAIL
    os.environ["TORCH_NCCL_ENABLE_MONITORING"]="1"
    
    size = num_gpus
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=isaac_training, args=(isaac_cfg, rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()