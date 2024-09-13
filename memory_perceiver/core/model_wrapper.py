import torch
from einops import repeat, rearrange

from memoryPerceiver.memory_perceiver.processing.vectorize import compute_fourier_encoding, fourier_encode
from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO


def create_multimodal_perceiver(cf, device):

    # Projection to return modalities, enhancement

    # input_parameters = {}
    # if cf.input_audio:
    #     input_parameters['audio'] = cf.query_dim - 2*cf.bands_fourier - cf.patch_audio
    # if cf.input_video:
    #     input_parameters['video'] =  cf.query_dim - 2*cf.bands_video_t - 2*2*cf.bands_video \
    #                                 - cf.patch_height*cf.patch_width*cf.channels
    # for part, params in cf.body_parts.items():
    #     input_parameters[part] = cf.query_dim - 2*cf.bands_fourier - params
    # for part, params in cf.body_outputs.items():
    #     input_parameters[part] = cf.query_dim - 2*cf.bands_fourier - params


    input_parameters = {}
    if cf.input_audio:
        input_parameters['audio'] = cf.patch_audio
    if cf.input_video:
        input_parameters['video'] =  cf.patch_height*cf.patch_width*cf.channels
    for part, params in cf.body_parts.items():
        input_parameters[part] = params

    output_parameters = dict(cf.body_outputs)
    if cf.output_audio:
        output_parameters.update({'audio': cf.patch_audio})
    for part, params in cf.body_outputs.items():
        output_parameters[part] = params

    # Instantiate model
        
    model = PerceiverIO(
            cf.depth,
            device=device,
            query_dim=cf.query_dim,
            latent_dim_head=cf.latent_dim_head,
            nb_latents = cf.nb_latents,
            nb_latents_mem = cf.nb_latents_mem,
            input_parameters = input_parameters,
            output_parameters = output_parameters
        )
    model = model.to(device)

    # Queries

    queries = []
    if cf.output_audio:
        audio_query = repeat( model.modality['audio'], '... -> n ...', n=cf.audio_future)
        audio_pos = fourier_encode(torch.linspace(-1, 1, cf.audio_future), cf.audio_future//2, num_bands=cf.query_dim//2).to(device)
        queries.append( (audio_query + audio_pos).to(device))
    for part, params in cf.body_outputs.items():
        part_query = repeat( model.modality[part], '... -> n ...', n=cf.frames_future)
        part_pos = fourier_encode(torch.linspace(-1, 1, cf.frames_future), cf.frames_future//2, num_bands=cf.query_dim//2).to(device)
        queries.append( (part_query + part_pos).to(device))
    model.queries=torch.cat(queries)

    return model

def input_embedding(model, data, device='cpu', 
                    height=8, width=8, patch_audio=16):
    b = len(next(iter(data.values())))

    fourier_encoding = compute_fourier_encoding(data, model.query_dim, device=device,
                                                height=height, width=width, patch_audio=patch_audio)
    multi_mod = torch.tensor([], device=device)
    for key, value in data.items():
        if key == 'video':
            video = rearrange(value, "b c t (h h1) (w w1) -> b (t h w) (h1 w1 c)", h1=height, w1=width)
            video_encodding = repeat(fourier_encoding['video'], '... -> b ...', b=b)
            x = model.out_embedding['video'](video) + video_encodding
            multi_mod = torch.cat([multi_mod, x], dim=1)
        elif key == 'audio':
            audio = rearrange(audio, "b (t t1) -> b t t1", t1=patch_audio)
            audio_encoding = repeat(fourier_encoding['audio'], '... -> b ...', b=b)
            x = model.out_embedding['audio'](audio) + audio_encoding
            multi_mod = torch.cat([multi_mod, x], dim=1)
        else:
            value_encoding = repeat(fourier_encoding[key], '... -> b ...', b=b)
            x = model.out_embedding[key](value) + value_encoding
            multi_mod = torch.cat([multi_mod, x], dim=1)
    return multi_mod



def output_projection(model, x, frames_future, audio_future):
    output = {}
    start_modality = 0
    for part in model.projections.keys():
        if part == 'audio':
            output['audio'] = rearrange(model.projections['audio'](
                                            x[:, :audio_future, :]), "b t t1 -> b (t t1)" )
            start_modality = audio_future
        else:
            output[part] = model.projections[part](
                                x[:, start_modality:start_modality+frames_future, :] )
            start_modality += frames_future
    return output

def reset_memories(memories, indexes=None):

    # Reset memory for new lifelong learning

    batch_size, depth, nb_latents_mem, query_dim = memories.shape
    if indexes is None:
        indexes=torch.arange(batch_size)
    with torch.no_grad():
        memories[indexes] *= 0

def copy_memories(layers_mem):
    device = layers_mem.device
    return layers_mem.clone().detach().to(device)

def create_memories(model, batch_size, device=torch.device('cpu')):
    layers_mem = torch.zeros(batch_size, model.depth, model.nb_latents_mem, model.query_dim,
                              device=device, requires_grad=True)
    return layers_mem

