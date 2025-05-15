from pathlib import Path

class IsaacCfg:

    experiment_name = "P_2.7_encoding"

    # Task
    task =  "Isaac-Humanoid-Direct-v0" # Isaac-Cartpole-Cam-Direct-v0
    agent_type = "Perceiver" # Simple Perceiver
    tutor_type = None # None Simple
    dim_observations = 75 # 75 4 5
    dim_actions = 21 # 21 1
    num_envs = 2048 # 1 54 128 864 2048 4096
    sim_duration = 32 # 1028 516 260
    seed = 68
    render = "rgb_array" # None human rgb_array

    # PPO
    frames = 1
    frames_future = 1
    dim_actions_ppo = min(2, frames_future)*dim_actions
    training_mode = 'Backward' # Backward TBPTT RTRL
    learning_rate = 1e-5 # 3e-4 1e-4 1e-5
    kl_threshold = 0.008 # 0.004 0.008 1e10
    scheduler = 'kl_div' # '' kl_div kl-warm-up_400 warm-up_100 warm-up_400
    range_lr = [1e-5, 1e-3] # [1e-5, 1e-2] [1e-5, 5e-3] [1e-5, 5e-4]
    reward_scale = 0.01
    start_step = 4
    c_entropy = 0.0 #5e-5 1e-4 # 0.01 to 0.05
    c_critic = 2.0   # 0.5 to 2 4
    c_kl_tutor = 0.0 # 0.2
    accumulation_steps = 32 # 2 4 6 8 16
    student_update_step = 1
    eps = 1e-9
    gamma = 0.99
    c_lambda = 0.95
    grad_norm = 0.5  # 0.1 to 1.0
    eps_clip = 0.2
    value_clip = 0.1 #float('inf')
    num_replay = 4
    update_probability = 0.5 #0.33 0.5 1

    # Model parameters
    depth = 1
    critic_depth = 1
    depth_f_state = 0
    query_dim = 256 # 32 128 256 512 832 # video > 136 , 
    hidden_dim = 256 # 32 128 256 for simple net
    latent_heads = 4 # 2 4 8
    latent_dim_head =  64 # 16 32 64
    nb_latents_mem = 0
    nb_latents = 8
    shared=True
    batch_size = num_envs
    # actions_variance = {'action_variance_' + str(i): 1 for i in range(dim_actions)}
    # body_outputs.update(actions_variance)

    # Model inputs and outputs
    inputs = ['obs', 'actions_star'] # obs audio video actions_star
    output_audio = False
    sample_audio_frame = 1920
    width = 448
    height = 256
    patch_audio = 16
    patch_width = 16
    patch_height = 16
    bands_fourier = 128
    bands_video = 32
    bands_video_t = 4
    channels = 3
    len_video_frame = 448 # 836 = 456*264/(12*12) # 1824 456*256/(8*8) # 448 448*256/(16*16)
    len_audio_frame = 120   # 1920/16
    sample_rate = frames * sample_audio_frame
    len_video = frames*len_video_frame
    len_audio = frames*len_audio_frame
    len_audio_raw = frames*sample_audio_frame
    video_future = frames_future*len_video_frame
    audio_future = frames_future*len_audio_frame 

    # Directories

    path_main = Path(__file__).parent
    path_experiment = path_main / 'experiments' / task / experiment_name
    path_weights = path_main / 'weights' / task / experiment_name 
    path_model = path_main / 'weights' / task / experiment_name / 'model.pt'
    path_model_tutor = path_main / 'weights' / task / experiment_name / 'model_tutor.pt'
    path_memories = path_main / 'weights' / task / experiment_name / 'mem'
    path_buffer = path_main / 'weights' / task / experiment_name / 'buffer'
    path_scaler = path_main / 'weights' / task / experiment_name / 'scaler.pt'
    path_logs = path_experiment / 'isaac.log'
    path_monitor = path_main / 'monitor'

isaac_cfg = IsaacCfg()