import torch
from einops import repeat, rearrange

from memoryPerceiver.memory_perceiver.processing.vectorize import fourier_encode
from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO

def create_multimodal_perceiver(cf, device):

    # Projection to return modalities, enhancement

    modality_parameters = {}
    modality_parameters['audio'] = cf.query_dim - 2*cf.bands_fourier - cf.patch_audio
    modality_parameters['video'] =  cf.query_dim - 2*cf.bands_video_t - 2*2*cf.bands_video \
                                  - cf.patch_height*cf.patch_width*cf.channels
    for part, params in cf.body_parts.items():
        modality_parameters[part] = cf.query_dim - 2*cf.bands_fourier - params
    for part, params in cf.body_outputs.items():
        modality_parameters[part] = cf.query_dim - 2*cf.bands_fourier - params

    projection_parameters = dict(cf.body_outputs)
    if cf.output_audio:
        projection_parameters.update({'audio': cf.patch_audio})

    # Input compression

    special_compression = []
    i0=0
    if 'video' in cf.inputs:
        special_compression += [i for i in range(cf.video_future)]
        i0 += cf.len_video
    if 'audio' in cf.inputs:
        special_compression += [i0 + i for i in range(cf.audio_future)]
        i0 += cf.len_audio
    for k in range(len(cf.body_parts)):
        special_compression += [i0 + k*cf.frames + i for i in range(cf.frames_future) ]

    # Instantiate model
        
    model = PerceiverIO(
            cf.depth,
            cf.query_dim,
            nb_latents = cf.nb_latents,
            mem_size = cf.mem_size,
            cmem_size = cf.cmem_size,
            nb_compressed = cf.nb_compressed,
            device=device,
            projection_parameters = projection_parameters,
            modality_parameters = modality_parameters,
            special_compression=special_compression
        )
    model = model.to(device)

    # Queries

    queries = []
    if cf.output_audio:
        audio_query = repeat( torch.cat([model.modality['audio'], torch.zeros(16, device=device)]), '... -> n ...', n=cf.audio_future)
        audio_pos = fourier_encode(torch.linspace(-1, 1, cf.audio_future), cf.audio_future/2, num_bands=128).to(device)
        queries.append(torch.cat([audio_query, audio_pos], dim=-1).to(device))
    for part, params in cf.body_outputs.items():
        part_query = repeat( torch.cat([model.modality[part], torch.zeros(params, device=device)]), '... -> n ...', n=cf.frames_future)
        part_pos = fourier_encode(torch.linspace(-1, 1, cf.frames_future), cf.frames_future/2, num_bands=128).to(device)
        queries.append(torch.cat([part_query, part_pos], dim=-1).to(device))
    model.queries=torch.cat(queries)

    return model

def output_projection(model, x, body_outputs, frames_future, audio_future):
    output = {}
    start_modality = 0
    if 'audio' in model.projections.keys():
        output['audio'] = rearrange(model.projections['audio'](
            x[:, :audio_future, :]), "b t t1 -> b (t t1)" )
        start_modality = audio_future
    for part in body_outputs.keys():
        output[part] = model.projections[part](
            x[:, start_modality:start_modality+frames_future, :] )
        start_modality += frames_future
    return output

def reset_memories(memories, indexes=None):

    # Reset memory for new lifelong learning

    batch_size, depth, mem_size, queries_dim = memories['layers_mem'].shape
    _, cmem_size, _ = memories['cross_cmem'].shape
    if indexes is None:
        indexes=torch.arange(batch_size)
    memories['cross_cmem'][indexes] *= 0
    memories['layers_mem'][indexes] *= 0
    memories['layers_cmem'][indexes] *= 0

def copy_memories(memories):
    device = memories['cross_cmem'].device
    cross_cmem = memories['cross_cmem'].clone().detach().to(device)
    layers_mem = [memories['layers_mem'].clone().detach().to(device) for layer_mem in layers_mem]
    layers_cmem = [memories['layers_cmem'].clone().detach().to(device) for layer_cmem in layers_cmem]
    return {'cross_cmem':cross_cmem, 'layers_mem':layers_mem, 'layers_cmem':layers_cmem}

def create_memories(model, batch_size, device=torch.device('cpu'), mem_size=1024, cmem_size=1024):
    depth, mem_size, cmem_size, queries_dim = model.depth, model.mem_size, model.cmem_size, model.queries_dim
    cross_cmem = torch.zeros(batch_size, cmem_size, queries_dim, device=device)
    layers_mem = torch.zeros(batch_size, depth//2, mem_size, queries_dim, device=device)
    layers_cmem = torch.zeros(batch_size, depth//2, cmem_size, queries_dim, device=device)
    return {'cross_cmem':cross_cmem, 'layers_mem':layers_mem, 'layers_cmem':layers_cmem}

