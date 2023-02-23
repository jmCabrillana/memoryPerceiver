import os
import os.path as osp
import sys
import numpy as np

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

class Config:

    # Sampling configuration

    fps = 32
    sepformer_rate = 8000
    sample_per_frame = 1920
    sample_rate = fps * sample_per_frame
    body_parts = {'observations':4, 'expectation':1}
    body_outputs = {'actions':1}
    seq_byte_size = 19012
    reward_byte_size = 21

    # Model dim configuration

    len_video_frame = 1824  # 456*256/(8*8)
    len_audio_frame = 120  # 1920/16
    query_dim = 832
    video_latent_dim = 256
    audio_latent_dim = 560

    

cf = Config()