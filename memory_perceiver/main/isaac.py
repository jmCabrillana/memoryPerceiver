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
from memoryPerceiver.memory_perceiver.core.model_net import create_memories, reset_memories
from memoryPerceiver.memory_perceiver.core.distributed import setup_distributed, average_gradients, scatter_dummy_src, gather_dummy_src
from memoryPerceiver.memory_perceiver.core.miscellaneous import custom_random, print_cuda_tensors, freeze_parameters
import memoryPerceiver.memory_perceiver.isaac.trainer as tr
from memoryPerceiver.memory_perceiver.isaac.scaler import RunningStandardScaler
from memoryPerceiver.memory_perceiver.isaac.plot import Plotter
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCriticSimple, ActorCriticPerceiver, create_agent
from memoryPerceiver.memory_perceiver.core.rtrl import  RTRL_partial, shuffle_split_indexes


def isaac_training(cf, rank, size):

    setup_distributed(rank, size)
    rank_sim = 0
    # rank_agents = [0] if size == 1 else list(range(1, size))
    rank_agents = list(range(0, size))
    group_agents = dist.new_group(rank_agents)
    device = torch.device(rank)
    nb_agents = len(rank_agents)
    assert cf.num_envs % nb_agents == 0
    agent_envs = cf.num_envs // nb_agents
    
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
        env_cfg = parse_env_cfg(cf.task, device=str(device), num_envs=cf.num_envs, use_fabric=False)
        env = gym.make(cf.task, cfg=env_cfg, render_mode=cf.render)
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
            memories = create_memories(model.actor, agent_envs, device=device)
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
        if rank == rank_agents[0]:
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
            slice_indexes = torch.zeros(agent_envs, len_slice_indexes, device=device, dtype=torch.long)
        dist.broadcast(slice_indexes, src=rank_sim)  
        if rank in rank_agents:
            slice_indexes = slice_indexes[rank-1]
            rtrl_grad = RTRL_partial(model_student, param_state_filter='actor.state_', num_parameters=70000,
                        chunk_size=128, slice_indexes=slice_indexes.tolist())        

    history_keys = cf.inputs + ['actions', 'actions_mean', 'actions_std', 'values', 'logprobs', 'rewards', 'returns', 'returns_targets', 
                    'advantages', 'birth_time', 'death_index']
    history = tr.init_history(history_keys, cf.sim_duration, agent_envs, cf.dim_actions_ppo, cf.dim_observations, \
                              cf.height, cf.width, cf.channels, device)
    history_sim = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, cf.dim_observations, \
                            cf.height, cf.width, cf.channels, device)
    next_state = tr.init_next_state(history_keys, agent_envs, cf.dim_actions_ppo, cf.dim_observations, \
                              cf.height, cf.width, cf.channels, device)
    next_state_sim = tr.init_next_state(history_keys, cf.num_envs, cf.dim_actions_ppo, cf.dim_observations, \
                            cf.height, cf.width, cf.channels, device)
    if rank in rank_agents:
        buffer = {'obs':[], 'video':[], 'audio':[], 'birth_time': [], 'returns': [torch.zeros(agent_envs, device=device)],
              'actions_star':[torch.zeros(agent_envs, cf.dim_actions, device=device)]}
    if rank in rank_agents and rank != rank_sim:
        for key in history_sim.keys():
            history_sim[key] = None
            next_state_sim[key] = None
    if rank not in rank_agents and rank == rank_sim:
        for key in history.keys():
            history[key] = None
            next_state[key] = None
    actions_star = torch.zeros(cf.frames_future, agent_envs, cf.dim_actions, device=device) if rank in rank_agents else None  
    actions_star_sim = torch.zeros(cf.frames_future, cf.num_envs, cf.dim_actions, device=device) if rank == rank_sim else None
    next_values = torch.zeros(agent_envs, device=device) if rank in rank_agents else None  
    next_values_sim = torch.zeros(cf.num_envs, device=device) if rank == rank_sim else None
    birth_time_sim = torch.zeros(cf.num_envs, device=device) if rank == rank_sim else None
    if rank in rank_agents:
        path_buffer_old = str(cf.path_buffer)+'_old'+str(rank)+'.pt'
        path_buffer_new = str(cf.path_buffer)+'_new'+str(rank)+'.pt'
        if not os.path.exists(path_buffer_new):
            torch.save(buffer, path_buffer_new)
    sim_count, sim_step = -1, 0
    sim_running = torch.tensor(False, device=rank)
    world_playing = torch.tensor(False, device=rank)
    if rank == rank_sim:
        sim_running.fill_(simulation_app.is_running())
    dist.broadcast(sim_running, src=rank_sim)    
    full_video = []
    r_grpo_len = 1000
    r_grpo_max = 1000
    r_grpo = torch.zeros(3, r_grpo_len, dtype=torch.float, device=rank) # m0, m1, m2 count of rewards by time
    c_entropy_bound = cf.c_entropy
    is_inference = True
    start_time = time.time()
    
    while sim_running:
        if rank == rank_sim:
            world_playing.fill_(env.unwrapped.sim.is_playing())
            sim_running.fill_(simulation_app.is_running())
        dist.broadcast(world_playing, src=0)
        dist.broadcast(sim_running, src=0)
        if world_playing:
            if sim_step % cf.sim_duration == 0:

                ### Plots

                if rank == rank_agents[0] and sim_count % (cf.num_replay+1) == 1 :
                    plotter.plot_all([('loss_actor', 64), ('loss_critic', 64), ('loss_entropy', 64), ('actions_gap', 64), 'returns'])

                ### Initialization

                sim_count += 1
                seed = sim_count // (cf.num_replay+1)
                is_inference =  sim_count % (cf.num_replay+1) == 0 # not is_inference
                dist.destroy_process_group(group_agents)
                group_agents = dist.new_group(rank_agents) # It is necessary to reset this group here to avoid duplicate gpu errors
                
                if rank == rank_sim:
                    # if sim_count % 200 == 0: # hard reset sometimes (to delete)
                    #     env.unwrapped._reset_idx(env_ids=None, hard_reset=True)
                    print(f"$$$ r{rank} MEAN REWARD: {history_sim['rewards'].mean().item()} $$$")
                    print(f"SEED: {seed}")
                    print(f"SIM_COUNT: {sim_count}")
                    is_inference_old = (sim_count-1) % (cf.num_replay+1) == 0
                    if is_inference_old: # end of inference, now training
                        print(f"Inference: {time.time() - start_time:.2f}")
                    else:
                        print(f"Training: {time.time() - start_time:.2f}")
                    start_time = time.time()

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
                        if rank == rank_agents[0] :
                            torch.save({'model': model_student.state_dict(), 'optimizer': optimizer.state_dict()}, cf.path_model)
                            if sim_count % 100 == 0:
                                torch.save({'model': model_student.state_dict(), 'optimizer': optimizer.state_dict()}, cf.path_weights / f'model_{sim_count}.pt')

                        history = tr.init_history(history_keys, cf.sim_duration, agent_envs, cf.dim_actions_ppo, cf.dim_observations, \
                                                  cf.height, cf.width, cf.channels, device)
                        buffer = torch.load(path_buffer_new, weights_only=True)
                        torch.save(buffer, path_buffer_old) # end of previous inference
                    if rank == rank_sim:
                        torch.save(value_scaler.state_dict(), cf.path_scaler)
                        if sim_count % 100 == 0:
                            torch.save(value_scaler.state_dict(), cf.path_weights / f'scaler_{sim_count}.pt')
                    kl_list = []
                    
                else:
                    if sim_count % (cf.num_replay+1) == 1:
                        if rank in rank_agents:
                            torch.save(buffer, path_buffer_new) # end of new inference

                        # Compute generalized advantage

                        gather_dummy_src(history['values'], history_sim['values'], rank, rank_agents, swap_axis=True)
                        gather_dummy_src(next_values, next_values_sim, rank, rank_agents)
                        if rank == rank_sim:
                            unscaled_next_values = value_scaler(next_values_sim, inverse=True) 
                            unscaled_values = value_scaler(history_sim['values'], inverse=True) 
                            history_sim['returns'] = tr.calculate_returns(history_sim['rewards'], history_sim['death_index'], gamma=cf.gamma)
                            tmp_advantages = tr.calculate_gae(history_sim['rewards'], unscaled_values, unscaled_next_values, history_sim['death_index'],
                                                            gamma=cf.gamma, gae_lambda=cf.c_lambda)
                            history_sim['values'] = value_scaler(unscaled_values, train=True) # != history_sim['values'] because update
                            history_sim['returns_targets'] = value_scaler(tmp_advantages + unscaled_values, train=True)
                            history_sim['advantages'] = (tmp_advantages - tmp_advantages.mean()) / (tmp_advantages.std() + 1e-8)  #dim=1 keepdim=True
                            # tr.calculate_grpo(r_grpo, history_sim['returns'], history_sim['birth_time'], sim_count_total)
                            # tr.print_detail_returns(history_sim, unscaled_values, tmp_advantages, value_scaler)
                            # print("$$$  SIM  $$$ ")
                            # tr.print_detail_env( history_sim)

                        for key in ['returns', 'values', 'returns_targets', 'advantages']:
                            scatter_dummy_src(history_sim[key], history[key], rank, rank_agents, swap_axis=True)
                        

                    dist.barrier()
                    if rank in rank_agents :
                        if isinstance(model, ActorCriticPerceiver):
                            memories = torch.load(path_mem, weights_only=True).detach()
                            if cf.training_mode == 'RTRL':
                                memories = rearrange(memories, 'b l h -> (b l h)')
                            memories_student = memories.clone().detach().requires_grad_(True)
                        buffer = torch.load(path_buffer_old, weights_only=True)
                        loss_cumul=torch.tensor(0.0, device=device)
                        backward_count=0
                        actor_losses, critic_losses, entropy_losses, kl_tutor_losses = [], [], [], []

                    if rank == rank_agents[0]:
                        plotter.add('returns', history['returns'][0, :].mean().item(), sim_count)
                        plotter.add('action/mean', history['actions'][:, :, 0].mean().item(), sim_count)
                        plotter.add('action/std', history['actions'][:, :, 0].std().item(), sim_count)
                        plotter.add('coeff/lr', lr, sim_count)
                        tr.print_detail_env(history)

                    
                sim_step = 0
            
            sim_count_total = sim_count*cf.sim_duration+sim_step

            ### Get agent observations and dones
            if is_inference:
                if rank == rank_sim:

                    # obs
                    all_obs = env.unwrapped._get_observations()
                    history_sim['obs'][sim_step] = all_obs['policy'].clone()    # (num_envs, d)
                    if 'video' in cf.inputs:
                        history_sim['video'][sim_step] = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                        # history_sim['video'][sim_step] = torch.zeros(num_envs, cf.height, cf.width, cf.channels, device=device)
                        # ts.save(history_sim['video'][sim_step][0], cf.path_monitor / 'live.jpg')
                    if cf.render is not None:
                        full_video.append(torch.tensor(env.render()))
                    
                    # dones
                    dones = env.unwrapped.last_dones.int().clone()
                    history_sim['birth_time'][sim_step] = birth_time_sim.clone() if sim_count_total > 0 else torch.zeros(cf.num_envs, device=device)
                    history_sim['birth_time'][sim_step, dones==1] = sim_count_total
                    birth_time_sim = history_sim['birth_time'][sim_step]
                    history_sim['death_index'][sim_step]=dones
                    history_sim['actions_star'][sim_step] = actions_star_sim[(sim_step-1) % cf.frames_future].clone()

                # Scatter for agent buffers
                for key in ['birth_time', 'death_index'] + cf.inputs:
                    scatter_dummy_src(tr.get_or_none(history_sim[key], sim_step), 
                                      tr.get_or_none(history[key], sim_step), rank, rank_agents)
                    if rank in rank_agents and key != 'death_index':
                        tr.buffer_update(buffer, key, history[key][sim_step], frames=cf.frames)   


            ### Agents act and train
            
            if sim_step % cf.frames_future == 0 and rank in rank_agents:

                ## Act
                if is_inference:
                    ## Inference, save history and memory update
                    
                    if isinstance(model, ActorCriticPerceiver):
                        dones_list = torch.nonzero(history['death_index'][sim_step].int()).flatten().tolist()
                        reset_memories(memories, indexes=dones_list, mem_dim=mem_dim)
                    mask = tr.create_mask(history['birth_time'][sim_step], sim_count_total, frames=cf.frames)
                    states=tr.batch_and_mask_states(tr.stack_buffer(buffer), agent_envs, cf, mask, device)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        model_outputs = model(states, memories, mask=mask)
                        actions, logprobs, values, new_memories = [model_outputs[key] for key in ['actions', 'logprobs', 'values', 'memories']]
                        if cf.training_mode != 'Backward':
                            memories = new_memories
                        if cf.tutor_type is not None:
                            model_tutor_outputs = model_tutor(states, None, mask=mask)
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
                    sim_step_rand=sim_step 
                    actions, actions_mean, actions_std, logprobs, values, birth_time, death_index = [history[key][sim_step_rand].to(device) \
                        for key in ['actions', 'actions_mean', 'actions_std', 'logprobs', 'values', 'birth_time', 'death_index']]
                    if isinstance(model, ActorCriticPerceiver):
                        dones_list = torch.nonzero(death_index.int()).flatten().tolist()
                        reset_memories(memories, indexes=dones_list, mem_dim=mem_dim)
                    last_state = {key: history[key][sim_step] for key in cf.inputs}
                    tr.buffers_update(buffer, last_state, frames=cf.frames)  
                    mask = tr.create_mask(birth_time, sim_count_total, frames=cf.frames)
                    states = tr.batch_and_mask_states(tr.stack_buffer(buffer), agent_envs, cf, mask, device)

                    # perform_update = ( custom_random(sim_count, sim_step_rand) <= cf.update_probability and \
                    #                     sim_step_rand <= cf.sim_duration-1 ) or cf.training_mode == 'TBPTT'
                    # perform_update = perform_update and seed >= cf.start_step
                    perform_update = seed >= cf.start_step
                    with autocast(device_type='cuda', dtype=torch.float16):
                        if perform_update:
                            returns_targets = history['returns_targets'][sim_step_rand].to(device)
                            advantages = history['advantages'][sim_step_rand].to(device)
                            if cf.training_mode in ['Backward', 'RTRL'] and isinstance(model, ActorCriticPerceiver):
                                memories_student = memories.clone().detach().requires_grad_(True)
                            model_student_outputs = model_student(states, memories_student, actions, mask)
                            new_actions_mean, new_actions_std, new_logprobs, new_values, entropies, new_memories_student = \
                                [model_student_outputs[key] for key in ['actions_mean', 'actions_std', 'logprobs', 'values', 'entropies', 'memories']]
                            if cf.training_mode == 'TBPTT':
                                memories_student = new_memories_student

                            # Loss
                            ratios = torch.exp(new_logprobs - logprobs)
                            surr1 = ratios * advantages
                            surr2 = torch.clamp(ratios, 1-cf.eps_clip, 1+cf.eps_clip) * advantages
                            loss_actor = -(torch.min(surr1, surr2))
                            loss_entropy = c_entropy_bound * entropies # cf.c_entropy * entropies.mean()
                            clipped_values = values + torch.clamp(new_values-values, -cf.value_clip, cf.value_clip)
                            loss_critic = cf.c_critic * F.mse_loss(clipped_values, returns_targets, reduction='none')
                            loss_kl_tutor = torch.tensor(0.0, device=device)
                            if cf.tutor_type is not None:
                                # loss_kl_tutor = F.mse_loss(new_actions_mean, actions_mean)
                                mu1, mu2, sig1, sig2 = new_actions_mean, actions_mean, new_actions_std, actions_std
                                loss_kl_tutor = cf.c_kl_tutor*0.5*( ( (mu2-mu1)/sig2 )**2 + (sig1/sig2)**2 - 2*torch.log(sig1) + 2*torch.log(sig2) - 1 ).mean(dim=1).mean()
                            loss = (loss_actor + loss_critic - loss_entropy + loss_kl_tutor) / cf.accumulation_steps
                            loss *= torch.bernoulli(torch.full((agent_envs,), cf.update_probability, device=device))
                            loss = loss.mean()
                            loss_cumul += loss
                            with torch.no_grad():
                                kl_divergence = ((ratios - 1) - (new_logprobs - logprobs)).mean()
                                kl_list.append(kl_divergence.item())
                            if kl_divergence > 3*cf.kl_threshold:
                                print("WARNING: kl_divergence > 3 * kl_threshold")
                                loss *= 0
                                loss_cumul *= 0
                        if cf.training_mode != 'Backward':
                            memories = model(states, memories)['memories']

                    if perform_update :

                        # Step optimizer
                        if cf.training_mode == 'Backward':
                            scaler.scale(loss/(cf.sim_duration*cf.update_probability)).backward()
                        if cf.training_mode == 'RTRL':
                            loss = scaler.scale(loss)
                            rtrl_grad.step(model_student, states, memories_student, loss, old_actions=actions)
                        backward_count += 1
                        # if backward_count % cf.accumulation_steps == 0 and backward_count != 0:
                        if backward_count > 0 and sim_step == cf.sim_duration-1:
                        # if backward_count % cf.accumulation_steps == 0 and backward_count != 0:
                            if cf.training_mode == 'TBPTT':
                                scaler.scale(loss_cumul).backward()
                                loss_cumul = torch.tensor(0.0, device=device, requires_grad=True)
                                if isinstance(model, ActorCriticPerceiver):
                                    memories_student = memories.clone().detach().requires_grad_(True)
                            if cf.training_mode == 'RTRL':
                                average_gradients(model_student, nb_agents, group_agents, param_filter='actor.out_')
                            else:
                                average_gradients(model_student, nb_agents, group_agents)  # ALL REDUCE on scaled loss
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model_student.parameters(), cf.grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                            # Reinitialize
                            backward_count = 0

                    if rank in rank_agents and perform_update :
                        actor_losses.append(loss_actor.mean().item())
                        critic_losses.append(loss_critic.mean().item())
                        entropy_losses.append(loss_entropy.mean().item())
                        kl_tutor_losses.append(loss_kl_tutor.mean().item())

            ### Step env,  close episode
            if is_inference:

                # Step environment with gathered action
                if sim_step % cf.frames_future == 0:
                    gather_dummy_src(actions_star, actions_star_sim, rank, rank_agents, swap_axis=True)
                if rank == rank_sim:
                    dones = env.unwrapped.last_dones.int().clone()
                    env.step(actions_star_sim[sim_step % cf.frames_future].clone())
                    history_sim['rewards'][sim_step] = cf.reward_scale*env.unwrapped._get_rewards()
                    # history_sim['returns'][sim_step] = (1 - dones) *  buffer_sim['returns'][-1] + cf.gamma * history_sim['rewards'][sim_step]
                    # tr.buffer_update(buffer, 'returns', history['returns'][sim_step], frames=cf.frames)
                    # r_grpo = tr.update_grpo(r_grpo, history['returns'][sim_step], sim_count_total, history['birth_time'][sim_step])
                    
                # Get next values at the last iteration
                
                if sim_step == cf.sim_duration-1:
                    next_sim_step = sim_step+1
                    
                    if rank == rank_sim:
                        all_obs = env.unwrapped._get_observations()
                        next_state_sim['obs'] = all_obs['policy'].clone()    # (num_envs, d)
                        if 'video' in cf.inputs:
                            history['video'] = all_obs['camera'].clone()  # (num_envs, 256, 456, 3)
                            # next_state['video'] = torch.zeros(num_envs, cf.frames, cf.height, cf.width, cf.channels, device=device)
                        dones = env.unwrapped.last_dones.int().clone()
                        next_state_sim['birth_time'] = birth_time_sim.clone()
                        next_state_sim['birth_time'][dones==1] = sim_count*cf.sim_duration+next_sim_step
                        next_state_sim['actions_star'] = actions_star_sim[(next_sim_step-1) % cf.frames_future].clone()
                    if rank in rank_agents:
                        next_buffer = deepcopy(buffer)
                    for key in ['birth_time'] + cf.inputs:
                        scatter_dummy_src(next_state_sim[key], next_state[key], rank, rank_agents)
                        if rank in rank_agents:
                            tr.buffer_update(next_buffer, key, next_state[key], frames=cf.frames)   
                        
                    if rank in rank_agents:
                        mask = tr.create_mask(next_state['birth_time'], sim_count*cf.sim_duration+next_sim_step, frames=cf.frames)
                        states=tr.batch_and_mask_states(tr.stack_buffer(next_buffer), agent_envs, cf, mask, device)
                        with autocast(device_type='cuda', dtype=torch.float16):
                            next_values = model(states, memories, mask=mask)['values']
                        next_values = next_values.to(torch.float32).clone().detach()
                        del(next_buffer)

                # video
                if sim_count % 100 == 0 and sim_step == cf.sim_duration-1 and rank == 0 and cf.render is not None :
                    torchvision.io.write_video( cf.path_monitor / ('test_'  + str(sim_count) + '_vid_.mp4'), 
                                                torch.stack(full_video), 8*cf.frames)
                    full_video = []
            else:

                # Adjust the learning rate after each training

                if sim_step == cf.sim_duration-1 and rank in rank_agents and seed >= cf.start_step \
                    and sim_count % (cf.num_replay + 1) == cf.num_replay:
                    
                    kl = torch.tensor(kl_list, device=device).mean()
                    dist.all_reduce(kl, op=dist.ReduceOp.SUM, group=group_agents)
                    kl /= nb_agents
                    if rank == rank_agents[0]:
                        plotter.add('loss/kl_divergence', kl.item(), sim_count)
                    if cf.scheduler == 'kl_div':
                        lr = tr.lr_kl_scheduler(lr, kl, kl_threshold=cf.kl_threshold, min_lr=cf.range_lr[0], max_lr=cf.range_lr[1])
                    if cf.scheduler.startswith('warm-up'):
                        lr = tr.lr_warm_up(seed, min_lr=cf.range_lr[0], max_lr=cf.range_lr[1], 
                                           steps_warm_up=int(cf.scheduler.split('_')[-1]))
                    if cf.scheduler.startswith('kl-warm-up'):
                        lr = tr.kl_scheduler_warm_up(seed, lr, kl, kl_threshold=cf.kl_threshold, min_lr=cf.range_lr[0], 
                                                     max_lr=cf.range_lr[1], steps_warm_up=int(cf.scheduler.split('_')[-1]))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # entropy lower bound

                    entropy = torch.tensor(entropy_losses, device=device).mean() / c_entropy_bound
                    dist.all_reduce(entropy, op=dist.ReduceOp.SUM, group=group_agents)
                    entropy /= nb_agents
                    target_gap = max(0, min(cf.c_entropy, cf.dim_actions_ppo - entropy)) 
                    c_entropy_bound = 0.9 * c_entropy_bound + 0.1 * target_gap

                    # Plot losses

                if rank == rank_agents[0] and sim_step == cf.sim_duration-1:
                    plotter.add('coeff/c_entropy', c_entropy_bound, sim_count)
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
    os.environ["TORCH_NCCL_ENABLE_MONITORING"]="0" # 0 1
    os.environ["CUDA_LAUNCH_BLOCKING"]="0" # 0 1 
    
    size = num_gpus
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=isaac_training, args=(isaac_cfg, rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()