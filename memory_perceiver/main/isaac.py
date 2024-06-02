from os.path import join, dirname
import os
import sys
import logging
from copy import deepcopy
from hydra import initialize, compose
from omegaconf import DictConfig
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

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

from memoryPerceiver.memory_perceiver.main.configuration import cf
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_memories, reset_memories
import memoryPerceiver.memory_perceiver.isaac.trainer as tr
from memoryPerceiver.memory_perceiver.isaac.plot import Plotter
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCritic, ActorCriticSimple, ActorCriticPerceiver

"""
log_path = '/isaac-sim/memoryPerceiver/memory_perceiver/main/outputs/isaac.log'
#log_file = open(log_path, 'w').close()
logging.basicConfig(filename=log_path, level=logging.DEBUG)
log_handler = logging.FileHandler(log_path, mode='a')
log_handler.setLevel(logging.DEBUG)

# Redirect stdout and stderr to the logging handler
sys.stdout = log_handler.stream
#sys.stderr = log_handler.stream
"""

def setup(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=size)

def average_gradients(model, size, group):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
        param.grad.data /= size

def scatter_dummy_src(tensor_batch, rank, size, swap_axes=False):
    #  split batch at 0 to 1, ..., size-1
    if rank == 0:
        tensor_list = list(torch.split(tensor_batch, len(tensor_batch)//size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        dst = torch.zeros_like(tensor_list[0], device=0)
    if rank > 0:
        tensor_list = None
        dst = tensor_batch
    dist.barrier()
    print(rank, "*")
    dist.scatter(dst, tensor_list, 0)

def gather_dummy_src(tensor, rank, size):
    #  gather and batch 1, ..., size-1 to 0
    if rank > 0:
        src = tensor
        tensor_list = None
    if rank == 0:
        tensor_list = list(torch.split(tensor, len(tensor)//size))
        tensor_list.insert(0, torch.zeros_like(tensor_list[0], device=0))
        src = torch.zeros_like(tensor_list[0], device=0)
        # tensor_list.pop(0)
        # tensor = torch.stack(tensor_list)
    dist.barrier()
    dist.gather(src, tensor_list, 0)
    

def isaac_training(cf, cfg: DictConfig, rank, size):

    setup(rank, size)
    group_agents = dist.new_group(list(range(1,size)))
    device = torch.device(rank)

    cfg_dict = omegaconf_to_dict(cfg)
    

    if rank == 0:
        print("rank ", rank, "before in env")
        # !!! Load isaac modules after the variables
        env = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.device_id, enable_viewport=cfg.enable_viewport)
        from omni.isaac.core.utils.torch.maths import set_seed
        import omni.usd
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.stage import open_stage, save_stage
        enable_extension("omni.isaac.sensor")
    if rank > 0:
        model = ActorCriticPerceiver(cf, rank)
        model_student = ActorCriticPerceiver(cf, rank)
        memories = create_memories(model.actor, cf.num_envs, device=device)
        last_memories = torch.zeros_like(memories['cross_cmem'], device=device)

    cf.num_envs = cf.num_envs//(size-1) if rank > 0 else cf.num_envs
    plotter = Plotter(titles=['loss_actor', 'loss_critic', 'returns'])
    history_keys = ['actions', 'values', 'logprobs', 'returns', 'rewards', 'advantages', 'death_index', 'generator']
    history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, device)
    previous_history = deepcopy(history)
    buffer = {'video':[], 'audio':[], 'obs':[], 'rewards':[], 'actions_star':[]}
    obs = torch.zeros(cf.num_envs, cf.dim_actions, device=rank)
    video = torch.zeros(cf.num_envs, cf.height, cf.width, cf.channels, device=rank, dtype=torch.uint8)
    reward = torch.zeros(cf.num_envs, 1, device=rank)
    actions_star = torch.zeros(cf.num_envs, cf.dim_actions, device=rank)

    if rank == 0:
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
        cfg_dict['seed'] = cfg.seed  # sets seed. if seed is -1 will pick a random one
        task = initialize_task(cfg_dict, env)
    if rank > 0:
        optimizer = torch.optim.Adam(model_student.parameters(), lr=cf.learning_rate)
        scaler = GradScaler()
    sim_count, sim_step = -1, 0
    is_inference=False  # Will switch to True immediately
    
    sim_running = torch.tensor(False, device=rank)
    world_playing = torch.tensor(False, device=rank)
    if rank == 0:
        sim_running.fill_(env._simulation_app.is_running())
    dist.broadcast(sim_running, src=0)    
    
    #while sim_running and sim_count < 6:
    while sim_running:
        if rank == 0:
            world_playing.fill_(env._world.is_playing())
            sim_running.fill_(env._simulation_app.is_running())
        dist.broadcast(world_playing, src=0)
        dist.broadcast(sim_running, src=0)
        if world_playing:
            if sim_step % cf.sim_duration == 0:
                
                ### Plots

                if sim_count % 6 == 1 and len(plotter.data['loss_actor'])>0:
                    plotter.plot_all([(plotter.data['loss_actor'], 160), (plotter.data['loss_critic'], 160), plotter.data['returns']])

                ### Initialization

                sim_count+=1
                is_inference = not is_inference
                if rank == 0:
                    env._world.reset(soft=False)
                
                
                if len(history['rewards'])>0:
                    print('$$$$$$$$$$$ CUMUL REWARD $$$$$$$$$$$')
                    print(history['rewards'].sum())
                if is_inference:
                    print('PLAY')
                    if rank > 0:
                        print("MEM: ", F.mse_loss(last_memories, memories['cross_cmem']).item())
                        if sim_count == 0 and os.path.exists(cf.path_model):
                            model.load_state_dict(torch.load(cf.path_model))
                            model_student.load_state_dict(torch.load(cf.path_model))
                        else:
                            if sim_count % cf.student_update_step*2 == 0:
                                model.load_state_dict(model_student.state_dict())
                        dist.barrier(group=group_agents)
                        if rank == 1:
                            torch.save(model_student.state_dict(), cf.path_model)
                        model.eval()
                else:
                    print('REPLAY')
                    scatter_dummy_src(history['death_index'], rank, size-1, swap_axes=True)
                    if rank > 0:
                        last_memories = memories['cross_cmem'].clone()
                        model_student.train() 
                        optimizer.zero_grad()
                    previous_history = deepcopy(history)
                    previous_history['returns'] = tr.calculate_returns(history['rewards'], history['death_index'], gamma=cf.c_gamma)
                    previous_history['advantages'] = tr.calculate_gae(history['rewards'], history['values'], 
                                                        history['death_index'], gamma=cf.c_gamma, gae_lambda=cf.c_lambda)
                    plotter.data['returns'].append(previous_history['returns'][1, :].mean().item())
                    
                if rank > 0:
                    reset_memories(memories)
                tr.set_seeds(sim_count//2)
                print("SEED: ", sim_count//2)
                full_video = []
                birth_time = torch.zeros(cf.num_envs, device=rank)
                for key in buffer.keys():
                    buffer[key]=[]
                history = tr.init_history(history_keys, cf.sim_duration, cf.num_envs, cf.dim_actions_ppo, device)
                backward_count= 0
                sim_step = 0

            print("rank ", rank, "step ", sim_step)
            ### Get agent observations 

            if rank == 0:
                all_obs = env._task.get_observations()['articulation_view']
                obs = all_obs['obs_buf'].clone()    # (num_envs, d)
                video = all_obs['video_buf'].clone()  # (num_envs, 256, 456, 3)
                #ts.save(video[0], '/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/live.jpg')
                #full_video.append(video.cpu())
                env._task.calculate_metrics()       
                reward=env._task.rew_buf.clone()    # (num_envs, 1)
                history['rewards'][sim_step] = reward
            #for x in [obs, video, reward]:
            scatter_dummy_src(obs, rank, size-1) 
            scatter_dummy_src(video, rank, size-1)
            scatter_dummy_src(reward, rank, size-1) 
            print(rank, reward)
            tr.update_buffers(buffer, {'obs': obs, 'video': video, 'rewards': reward}, frames=cf.frames)

            ### Reset memory at death and record 

            reset_indexes = torch.zeros(cf.num_envs, device=rank)
            if rank == 0:
                for agent_i in range(cf.num_envs):
                    if env._task.reset_buf[agent_i].item() == 1:
                        reset_indexes[agent_i] = 1
                        birth_time[agent_i] = sim_step

                        # Update death dictionnary during inference

                        if is_inference:
                            history['death_index'][sim_step, agent_i]=1
            print("R ", rank, "A ")
            scatter_dummy_src(reset_indexes, rank, size-1) 
            print("R ", rank, "B ")
            scatter_dummy_src(birth_time, rank, size-1) 
            print("R ", rank, "C ")
            reset_indexes_list = torch.nonzero(reset_indexes) \
                                    .flatten().tolist()
            if rank > 0:
                reset_memories(memories, indexes=reset_indexes_list)

            print("rank ", rank, "SC ")

            ### Agent
            
            
            if len(buffer['actions_star'])==0 :
                if rank > 0:

                    ## Process inputs

                    states={}
                    mask = tr.create_mask(birth_time, sim_step, frames=cf.frames)
                    states['observations'] = tr.buffer_padding(torch.stack(buffer['obs']), [cf.dim_observations],
                                                                num_envs=cf.num_envs, frames=cf.frames)
                    states['observations'][mask] *= 0
                    states['video'] = torch.zeros(cf.num_envs, cf.frames, cf.height, cf.width, 3, device=device)
                    # states['video'] = tr.buffer_padding(torch.stack(buffer['video']), [cf.height, cf.width, 3], num_envs=cf.num_envs, frames=cf.frames)
                    states['video'][mask] *= 0
                    states['video'] = rearrange(states['video'], 'b t h w c -> b c t h w')
                    states['audio'] = torch.zeros(cf.num_envs, cf.len_audio_raw, device=device)

                    ## Inference, save history and memory update

                    if is_inference:
                        history['generator'][sim_step, rank] = torch.cuda.get_rng_state(rank).clone() 
                        with autocast(device_type='cuda', dtype=torch.float16):
                            actions, logprobs, values, memories = model(states, memories)   
                        history['actions'][sim_step] = actions  
                        history['logprobs'][sim_step] = logprobs 
                        history['values'][sim_step] = values     
                    else:
                        actions, logprobs, values = [previous_history[key][sim_step].to(device) \
                                                    for key in ['actions', 'logprobs', 'values']]
                                        

                    ## Backward pass

                    if not is_inference:

                        with autocast(device_type='cuda', dtype=torch.float16):
                            returns = previous_history['returns'][sim_step].to(device)
                            advantages = previous_history['advantages'][sim_step].to(device)
                            new_logprobs, new_values, entropies, _ = model_student(states, memories, actions)
                            torch.cuda.set_rng_state(previous_history['generator'][sim_step, rank], device=device)
                            _, _, _, memories = model(states, memories) 
                            

                            # Loss
                            ratios = torch.exp(new_logprobs - logprobs)
                            surr1 = ratios * advantages
                            surr2 = torch.clamp(ratios, 1-cf.eps_clip, 1+cf.eps_clip) * advantages
                            loss_actor = -(torch.min(surr1, surr2)).mean()
                            loss_entropy = -cf.c_entropy * entropies.mean()
                            loss_critic = cf.c_critic * F.mse_loss(new_values, returns)
                            loss =  (loss_actor + loss_critic + loss_entropy) / cf.accumulation_steps
                        scaler.scale(loss).backward()
                        backward_count += 1


                        # Step optimizer

                        if backward_count % cf.accumulation_steps == 0 and backward_count != 0:
                            scaler.unscale_(optimizer)
                            average_gradients(model, size-1, group_agents) # ALL REDUCE on scaled loss
                            torch.nn.utils.clip_grad_norm_(model_student.parameters(), cf.c_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                            # Reinitialize
                            backward_count = 0


                        plotter.data['loss_actor'].append(loss_actor.item())
                        plotter.data['loss_critic'].append(loss_critic.item())

                    # Update buffers

                    actions_star = model.instanciate_actions(actions, dt=cf.frames_future)
                gather_dummy_src(actions_star, 0, size-1)
                print("rank: ", rank, "gather ")
                buffer['actions_star'] += list(torch.unbind( actions_star ))


            # Act(num_envs, 1)

            if rank == 0:
                if(len(buffer['actions_star'])>0):
                    action = buffer['actions_star'].pop(0)
                    action = torch.clamp(action, min=-1, max=1)
                else:
                    action = torch.dzeros(cf.num_envs, 1)

                # Step environment

                if sim_step == 0:
                    env._task.post_reset()
                env._task.pre_physics_step(action.to(task.device))
                env._world.step(render=cf.render)
                env.sim_frame_count += 1
                sim_step+=1
                env._task.post_physics_step()

                if sim_count % 10 == 0 and env.sim_frame_count % cf.sim_duration == 256 and rank == 1:
                    torchvision.io.write_video('/isaac-sim/memoryPerceiver/memory_perceiver/main/monitor/test_'  + str(sim_count) + '_vid_.mp4', 
                                                torch.stack(full_video)[:,1], 8*cf.frames)
        else:
            if rank == 0:
                env._world.step(render=cf.render)

    if rank == 0:
        env._simulation_app.close()
    dist.destroy_process_group()


if __name__ == "__main__":

    log_path = '/isaac-sim/memoryPerceiver/memory_perceiver/main/outputs/isaac.log'
    sys.stdout = open(log_path, 'w')

    num_gpus=torch.cuda.device_count()
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with initialize(config_path="../../OmniIsaacGymEnvs/omniisaacgymenvs/cfg"):
        cfg = compose(config_name="config", overrides=["task=Humanoid", "headless=True", "enable_viewport=True",
                        "num_envs="+str(cf.num_envs), "imgWidth="+str(cf.width), "imgHeight="+str(cf.height)])

    size = num_gpus
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=isaac_training, args=(cf, cfg, rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()