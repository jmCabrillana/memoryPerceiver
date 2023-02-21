# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    save_step = 160
    obs_buffer = []
    reward_buffer = []
    action_buffer = []
    while env._simulation_app.is_running():
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                env._world.reset(soft=True)

            # 'cartpole_view' 'obs_buf' 512, 4
            obs = env._task.get_observations()
            reward = env._task.calculate_metrics()
            print('obs: ', obs, '\n', '$$$', '\n') 

            
            # obs_buffer.append(obs)
            # if(len(obs_buffer)>32):
                #  obs_buffer.pop(0)
            # reward_buffer.append(reward)
            # if(len(reward_buffer)>32):
                #  reward_buffer.pop(0)
            # media['video'] = default_video
            # media['audio'] = default_audio
            #
            # if env.sim_frame_count % save_step == 0 and len(obs_buffer) >= save_step:
            # Save mem body, audio, video


            # if(len(action_buffer) == 0 and len(obs_buffer) >= 8):
                # obs.append(get camera view())
                # obs.append(audio)
                # with torch.no_grad():
                    # action, voice = memory_perceiver(obs_buffer[-8:])
                # action_buffer.append(action)
            # if(len(action_buffer)>0):
                # action = action_buffer.pop(0)
            # else:
                # action = torch.randn(512, 1)


            action = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
            # 512, 1
            print('action: ', action.size(), '\n', '$$$', '\n') 
            env._task.pre_physics_step(action)
            env._world.step(render=render)
            env.sim_frame_count += 1
            env._task.post_physics_step()
        else:
            env._world.step(render=render)

    env._simulation_app.close()

if __name__ == '__main__':
    parse_hydra_configs()
