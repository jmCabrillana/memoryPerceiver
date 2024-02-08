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

    # Sampling and input/output configuration

    frames = 32
    sample_per_frame = 1920
    sample_rate = frames * sample_per_frame
    frames_future = 8
    len_video_frame = 1824  # 456*256/(8*8)
    len_audio_frame = 120   # 1920/16

    # Model dimension configuration

    query_dim = 832
    video_latent_dim = 256
    audio_latent_dim = 560
    depth = 2
    batch_size = 2
    body_parts = {'observations': 4} #, 'returns':1, 'past_actions':1}
    body_outputs = {'actions': 1}
    has_video = True
    has_audio = True

    # Resulting total length (input / output)

    len_video = frames*len_video_frame
    len_audio = frames*len_audio_frame
    len_audio_raw = len_audio*sample_per_frame
    video_future = frames_future*len_video_frame
    audio_future = frames_future*len_audio_frame   

cf = Config()