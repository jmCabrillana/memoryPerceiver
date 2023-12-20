import torch
from einops import repeat, rearrange

from memoryPerceiver.memory_perceiver.processing.vectorize import fourier_encode

from model_net import PerceiverIO

def create_multimodal_perceiver(cf, device):

    # Projection to return modalities, enhancement

    modality_parameters = {}
    modality_parameters['audio'] = cf.audio_latent_dim
    modality_parameters['video'] = cf.video_latent_dim
    for part, params in cf.body_parts.items():
        modality_parameters[part] = cf.query_dim - 2*128 - params
    for part, params in cf.body_outputs.items():
        modality_parameters[part] = cf.query_dim - 2*128 - params

    projection_parameters = dict(cf.body_outputs)
    projection_parameters.update({'audio': 16})

    # Input compression

    special_compression = [i for i in range(cf.video_future)] + [i for i in range(cf.len_video, cf.len_video+cf.audio_future)]
    for k in range(len(cf.body_parts)):
        special_compression += [ cf.len_video + cf.len_audio + k*cf.frames + i for i in range(cf.frames_future) ]

    # Instantiate model
        
    model = PerceiverIO(
            cf.depth,
            cf.query_dim,
            special_compression = special_compression,
            device=device,
            projection_parameters = projection_parameters,
            modality_parameters = modality_parameters,
            max_batch_size = cf.batch_size
        )
    model = model.to(device)

    # Queries

    queries = []
    audio_query = repeat( torch.cat([model.modality['audio'], torch.zeros(16, device=device)]), '... -> n ...', n=cf.audio_future)
    audio_pos = fourier_encode(torch.linspace(-1, 1, cf.audio_future), cf.audio_future/2, num_bands=128).to(device)
    queries.append(torch.cat([audio_query, audio_pos], dim=-1).to(device))
    for part, params in cf.body_outputs.items():
        part_query = repeat( torch.cat([model.modality[part], torch.zeros(params, device=device)]), '... -> n ...', n=cf.frames_future)
        part_pos = fourier_encode(torch.linspace(-1, 1, cf.frames_future), cf.frames_future/2, num_bands=128).to(device)
        queries.append(torch.cat([part_query, part_pos], dim=-1).to(device))
    model.queries=torch.cat(queries)

    return model

def output_projection(model, x, cf):
    output = {}
    output['audio'] = rearrange(model.projections['audio'](
        x[:, :cf.audio_future, :]), "b t t1 -> b (t t1)" )
    start_modality = cf.audio_future
    for part in cf.body_outputs.keys():
        output[part] = model.projections[part](
            x[:, start_modality:start_modality+cf.frames_future, :] )
        start_modality += cf.frames_future
    return output
