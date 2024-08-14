" Isaac"

class Config:

    experiment_name = "h_e0.0"

    # Model inputs and outputs

    frames = 2
    sample_per_frame = 1920
    frames_future = 2
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

    # PPO
    task_name = "Humanoid" # Humanoid Cartpole 
    agent_type = "Simple" # Perceiver Simple
    dim_observations = 87 # 87 4
    dim_actions = 21 # 21 1
    dim_actions_ppo = frames_future*dim_actions # *2
    sim_duration = 20 #1028 #260
    num_envs = 18
    learning_rate = 1e-4 # 1e-4
    accumulation_steps = 4 #2
    c_entropy = 0.05  # 0.01 to 0.05
    c_critic = 0.4       # 0.5 to 2
    student_update_step = 1
    eps = 1e-9
    c_gamma = 0.99
    c_lambda = 0.95
    c_grad_norm = 0.1  # 0.1 to 1.0
    eps_clip = 0.2
    n_replay = 4
    c_update_probability = 0.1
    render = True

    # Model dimension configuration

    critic_layers = 4
    query_dim = 832
    depth = 2
    batch_size = num_envs
    body_parts =  {'observation_' + str(i): 1 for i in range(dim_observations)}
    body_outputs = {'action_' + str(i): 1 for i in range(dim_actions)}
    actions_variance = {'action_variance_' + str(i): 1 for i in range(dim_actions)}
    body_outputs.update(actions_variance)
    body_outputs.update({'critic':1})
    has_video = True
    has_audio = True
    path_main = '/isaac-sim/memoryPerceiver/memory_perceiver/main/'
    path_model = path_main+'weights/'+task_name+'.pt'
    path_logs = path_main+'outputs/isaac.log'


cf = Config()