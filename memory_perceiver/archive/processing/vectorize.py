import torch
from math import pi
from einops import rearrange, repeat

def fourier_encode(x, max_freq, num_bands = 4):
    device, dtype = x.device, x.dtype
    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    pulse = torch.einsum('i, j -> i j', x, scales) * pi
    encoding = torch.cat([pulse.sin(), pulse.cos()], dim = -1)
    return encoding


def compute_fourier_encoding(body_parts, device='cpu', query_dim=800, frames=1, height=64, width=64, audio=160,
                    patch_height=8, patch_width=8, patch_audio=16, bands_video=64, bands_video_t=64):
    
    fourier_encoding = {}

    #=======#
    # Video Encoding
    #=======#

    if 'video' in body_parts.keys():
        t, h0, w0 = frames, height, width
        h1, w1 = patch_height, patch_width
        h, w =  int(h0/h1), int(w0/w1)
        
        # Width position encoding

        pos = fourier_encode(torch.linspace(-1, 1, w), w/2, num_bands=bands_video).to(device)
        video_pos = repeat(pos, '... -> t h ...', t=t, h=h)

        # Height position encoding

        video_pos = rearrange(video_pos, "t h w d -> w t h d")
        pos = fourier_encode(torch.linspace(-1, 1, h), h/2, num_bands=bands_video).to(device)
        pos = repeat(pos, '... -> w t ...', w=w, t=t)
        video_pos = torch.cat([video_pos, pos], dim=-1)

        # Time axis position encoding

        video_pos = rearrange(video_pos, "w t h d -> h w t d")
        pos = fourier_encode(torch.linspace(-1, 1, t), t/2, num_bands=bands_video_t).to(device)
        pos = repeat(pos, '... -> h w ...', h=h, w=w)
        padding = torch.zeros(h, w, t, query_dim-4*bands_video-2*bands_video_t, device=device)
        video_pos = torch.cat([video_pos, pos, padding], dim=-1)
        fourier_encoding['video'] = video_pos

    #=======#
    # Audio Encoding
    #=======#

    if 'audio' in body_parts.keys():
        t0 = frames 
        t1 = patch_audio
        t =  int(t0/t1)

        # Time to depth

        audio = rearrange(audio, "b (t t1) -> b t t1", t1=t1)

        # Time axis postion encoding

        pos = fourier_encode(torch.linspace(-1, 1, t), t/2, num_bands=query_dim//2).to(device)
        fourier_encoding['audio'] = pos


    #=======#
    # Body Encoding
    #=======#

    for part in body_parts.keys():
        if part not in ['audio', 'video']:

            t = frames

            # Time axis postion encoding

            pos = fourier_encode(torch.linspace(-1, 1, t), t/2, num_bands=query_dim//2).to(device)
            fourier_encoding[part] = pos

    #=======#
    # Multi-modal result
    #=======#

    return fourier_encoding
