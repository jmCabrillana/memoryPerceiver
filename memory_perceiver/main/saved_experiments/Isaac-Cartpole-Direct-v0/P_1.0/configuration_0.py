" Isaac"

class Config:

    experiment_name = "P_1.0"

    # Task
    task = "Isaac-Cartpole-Direct-v0" # Humanoid Cartpole 
    agent_type = "Perceiver" # Simple Perceiver
    dim_observations = 4 # 87 4
    dim_actions = 1 # 21 1
    num_envs = 128 # 54 216 864
    sim_duration = 32 # 1028 516 260
    seed = 68
    render = False

    # PPO
    frames = 1
    frames_future = 1
    dim_actions_ppo = min(2, frames_future)*dim_actions
    training_mode = 'Backward' # Backward TBPTT RTRL
    learning_rate = 1e-5 # 3e-4 1e-4 1e-5
    kl_threshold = 0.008
    kl_scheduler = True
    range_lr = [1e-5, 1e-2]
    entropy = 0.00 # 0.01 to 0.05
    critic = 2   # 0.5 to 2 4
    accumulation_steps = 6 # 2 4
    student_update_step = 1
    eps = 1e-9
    gamma = 0.9
    c_lambda = 0.95
    grad_norm = 1.0  # 0.1 to 1.0
    eps_clip = 0.2
    value_clip = 0.2 #float('inf')
    num_replay = 1
    update_probability = 0.5 #0.33

    # Model parameters
    depth = 1
    critic_depth = 2
    depth_f_state = 0
    query_dim = 512 # 24 512 832 # video > 136 , 
    hidden_dim = 32 # for simple net
    latent_heads = 8 # 2 8
    latent_dim_head =  64 #16  64
    nb_latents_mem = 1
    nb_latents = 128
    shared=True
    batch_size = num_envs
    body_parts =  {'observation_' + str(i): 1 for i in range(dim_observations)}
    body_outputs = {'action_' + str(i): 1 for i in range(dim_actions)}
    actions_variance = {'action_variance_' + str(i): 1 for i in range(dim_actions)}
    body_outputs.update(actions_variance)
    body_outputs.update({'critic':1})

    # Model inputs and outputs
    input_audio = False
    input_video = False
    output_audio = False
    sample_audio_frame = 1920
    width = 456
    height = 264
    patch_audio = 16
    patch_width = 12
    patch_height = 12
    bands_fourier = 128
    bands_video = 32
    bands_video_t = 4
    channels = 3
    len_video_frame = 836 # 836 = 456*264/(12*12) # 1824 456*256/(8*8) # 448 448*256/(16*16)
    len_audio_frame = 120   # 1920/16
    sample_rate = frames * sample_audio_frame
    len_video = frames*len_video_frame
    len_audio = frames*len_audio_frame
    len_audio_raw = frames*sample_audio_frame
    video_future = frames_future*len_video_frame
    audio_future = frames_future*len_audio_frame 

    # Directories

    path_main = '/isaac-sim/memoryPerceiver/memory_perceiver/main/'
    path_experiment = path_main+'experiments/'+task+'/'+experiment_name+'/'
    path_model = path_main+'weights/'+task+'_'+experiment_name+'.pt'
    path_memories = path_main+'weights/'+task+'_'+experiment_name+'_mem_'
    path_scaler = path_main+'weights/'+task+'_'+experiment_name+'_scaler.pt'
    path_logs = path_experiment+'isaac.log'


cf = Config()