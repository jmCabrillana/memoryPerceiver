" Isaac"

class Config:

    experiment_name = "h_e0.0"
    experiment_iteration=0

    # Model inputs and outputs

    frames = 1
    sample_per_frame = 1920
    frames_future = 1
    width = 456
    height = 264
    patch_audio = 16
    patch_width = 12
    patch_height = 12
    bands_fourier = 128
    bands_video = 32
    bands_video_t = 4
    channels = 3
    len_video_frame = 836 # 836 = 456*264/(12*12) # 1824 # 456*256/(8*8)
    len_audio_frame = 120   # 1920/16
    sample_rate = frames * sample_per_frame
    len_video = frames*len_video_frame
    len_audio = frames*len_audio_frame
    len_audio_raw = frames*sample_per_frame
    video_future = frames_future*len_video_frame
    audio_future = frames_future*len_audio_frame 

    # Task
    task = "Isaac-Cartpole-Direct-v0" # Humanoid Cartpole 
    agent_type = "Simple" # Perceiver Simple
    dim_observations = 4 # 87 4
    dim_actions = 1 # 21 1
    dim_actions_ppo = min(2, frames_future)*dim_actions
    sim_duration = 516 # 1028 516 260
    num_envs = 2048 # 54 216 864
    seed = 68
    render = False

    # PPO
    learning_rate = 3e-4 # 1e-4
    accumulation_steps = 1 # 2 4
    entropy = 0.00  # 0.01 to 0.05
    critic = 4      # 0.5 to 2 4
    student_update_step = 1
    eps = 1e-9
    gamma = 0.99
    c_lambda = 0.95
    grad_norm = 1.0  # 0.1 to 1.0
    eps_clip = 0.2
    num_replay = 1
    update_probability = 0.99 #0.33

    # Model dimension configuration

    critic_layers = 4
    query_dim = 832
    depth = 2
    shared=True
    batch_size = num_envs
    body_parts =  {'observation_' + str(i): 1 for i in range(dim_observations)}
    body_outputs = {'action_' + str(i): 1 for i in range(dim_actions)}
    actions_variance = {'action_variance_' + str(i): 1 for i in range(dim_actions)}
    body_outputs.update(actions_variance)
    body_outputs.update({'critic':1})
    has_video = True
    has_audio = True
    path_main = '/isaac-sim/memoryPerceiver/memory_perceiver/main/'
    path_experiment = path_main+'experiments/'+task+'/'+experiment_name+'/'
    path_model = path_main+'weights/'+task+'.pt'
    path_optimizer = path_main+'weights/'+task+'_optimizer.pt'
    # path_logs = path_main+'outputs/isaac.log'


cf = Config()