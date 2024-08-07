# class Config:

#     # Sampling configuration

#     fps = 32
#     sepformer_rate = 8000
#     sample_per_frame = 1920
#     sample_rate = fps * sample_per_frame
#     body_parts = {'smplx_root_pose':3, 'smplx_body_pose':63, 'smplx_lhand_pose':45, 'smplx_rhand_pose':45, 'smplx_jaw_pose':3, 
#               'smplx_expr':10, 'cam_trans':3, 'bbox':4}
#     body_parts_extended = {'smplx_root_pose':3, 'smplx_body_pose':63, 'smplx_lhand_pose':45, 'smplx_rhand_pose':45, 'smplx_jaw_pose':3, 
#               'smplx_expr':10, 'cam_trans':3, 'bbox':4, 'smplx_shape':10}
#     seq_byte_size = 19012
#     reward_byte_size = 21

#     # Model dim configuration

#     len_video_frame = 1824  # 456*256/(8*8)
#     len_audio_frame = 120  # 1920/16
#     query_dim = 832
#     video_latent_dim = 256
#     audio_latent_dim = 560

    

# cf = Config()

" Isaac"

class Config:

    # Input / Outputs

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
    dim_observations = 87
    dim_actions = 21
    dim_actions_ppo = 2*21 # *2
    sim_duration =  260 #260
    num_envs = 12
    learning_rate = 1e-4
    accumulation_steps = 1 #2
    c_entropy = 0.05   # 0.01 to 0.05
    c_critic=0.2       # 0.5 to 2
    student_update_step = 1
    eps=1e-9
    c_gamma=0.99
    c_lambda=0.95
    c_grad_norm=0.1  # 0.1 to 1.0
    eps_clip=0.2
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
    has_video = True
    has_audio = True
    path_model = '/isaac-sim/memoryPerceiver/memory_perceiver/main/weights/humanoid.pt'


cf = Config()