import torch
from math import pi
from einops import rearrange, repeat

def fourier_encode(x, max_freq, num_bands = 4):
    device, dtype = x.device, x.dtype
    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    pulse = torch.einsum('i, j -> i j', x, scales) * pi
    encoding = torch.cat([pulse.sin(), pulse.cos()], dim = -1)
    return encoding



def input_to_tensor(data, latents, body_parts=None, device='cpu', 
                    height=8, width=8, bands_video=64, bands_video_t=64):
    
    # Set everything to the same device:

    for key, latent in latents.items():
        latents[key]=latent.to(device)
    for key, modality in data.items():
        data[key]=modality.to(device)
    multi_mod = torch.tensor([], device=device)

    #=======#
    # Video Encoding
    #=======#

    if 'video' in data:
        video = data['video']
        b, c, t, h0, w0 = video.shape # [colors, time, height, width]
        h1, w1 = height, width
        h, w =  int(h0/h1), int(w0/w1)
        
        # Space to depth

        video = rearrange(video, "b c t (h h1) (w w1) -> b t h w (h1 w1 c)", h1=h1, w1=w1)

        # Width position encoding

        pos = fourier_encode(torch.linspace(-1, 1, w), w/2, num_bands=bands_video).to(device)
        pos = repeat(pos, '... -> b t h ...', b=b, t=t, h=h)
        video = torch.cat([video, pos], dim=-1)

        # Height position encoding

        video = rearrange(video, "b t h w d -> b w t h d")
        pos = fourier_encode(torch.linspace(-1, 1, h), h/2, num_bands=bands_video).to(device)
        pos = repeat(pos, '... -> b w t ...', b=b, w=w, t=t)
        video = torch.cat([video, pos], dim=-1)

        # Time axis position encoding

        video = rearrange(video, "b w t h d -> b h w t d")
        pos = fourier_encode(torch.linspace(-1, 1, t), t/2, num_bands=bands_video_t).to(device)
        pos = repeat(pos, '... -> b h w ...', b=b, h=h, w=w)
        video = torch.cat([video, pos], dim=-1)

        # Add encoding

        latents_video = repeat(latents['video'], '... -> b h w t ...', b=b, h=h, w=w, t=t).to(device)
        video = torch.cat([video, latents_video], dim=-1)

        # Back to normal dimension order and flatten

        video = rearrange(video, "b h w t d -> b (t h w) d")
        multi_mod = torch.cat([multi_mod, video], dim=1)

    #=======#
    # Audio Encoding
    #=======#

    if 'audio' in data:
        audio = data['audio']
        b, t0 = audio.shape 
        t1 = 16
        t =  int(t0/t1)

        # Time to depth

        audio = rearrange(audio, "b (t t1) -> b t t1", t1=t1)

        # Time axis postion encoding

        pos = fourier_encode(torch.linspace(-1, 1, t), t/2, num_bands=128).to(device)
        pos = repeat(pos, '... -> b ...', b=b)
        audio = torch.cat([audio, pos], dim=-1)

        # Add encoding

        latents_audio = repeat(latents['audio'], '... -> b t ...', b=b, t=t).to(device)
        audio = torch.cat([audio, latents_audio], dim=-1)
        multi_mod = torch.cat([multi_mod, audio], dim=1)


    #=======#
    # Body Encoding
    #=======#

    # out['smplx_root_pose'], out['smplx_body_pose'], out['smplx_lhand_pose'],
    #                  out['smplx_rhand_pose'], out['smplx_jaw_pose'], out['smplx_shape'], 
    #                  out['smplx_expr'], out['cam_trans'], out['bbox']

    if body_parts is not None:
        body = torch.tensor([], device=device)
        for part in body_parts.keys():

            b, t, _ = data[part].shape 

            # Time axis postion encoding

            pos = fourier_encode(torch.linspace(-1, 1, t), t/2, num_bands=128).to(device)
            pos = repeat(pos, '... -> b ...', b=b)
            body_part = torch.cat([data[part], pos], dim=-1)

            # Add encoding

            part_encodings = repeat(latents[part], '... -> b t ...', b=b, t=t).to(device)
            body_part = torch.cat([body_part, part_encodings], dim=-1)
            body = torch.cat([body, body_part], dim=1)
        multi_mod = torch.cat([multi_mod, body], dim=1)
        

    #=======#
    # Multi-modal result
    #=======#

    return multi_mod