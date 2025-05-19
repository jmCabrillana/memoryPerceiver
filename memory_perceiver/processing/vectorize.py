import torch
from math import pi
from einops import rearrange, repeat

def fourier_encode(x, max_freq, num_bands = 4):
    device, dtype = x.device, x.dtype
    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    pulse = torch.einsum('i, j -> i j', x, scales) * pi
    encoding = torch.cat([pulse.sin(), pulse.cos()], dim = -1)
    return encoding

def fourier_encode_vector(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

def patchify(data, height=8, width=8, patch_audio=16):
    for key, value in data.items():
        if key == 'video':
            data['video'] = rearrange(value, "b t (h h1) (w w1) c -> b t h w (h1 w1 c)", h1=height, w1=width)
        elif key == 'audio':
            data['audio'] = rearrange(value, "b (t t1) -> b t t1", t1=patch_audio)

