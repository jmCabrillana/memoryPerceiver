from tqdm import tqdm
import os
from os.path import join
import gc
import sys
import time
from functools import wraps
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import HTML

import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torchaudio
from pytorchvideo.data.encoded_video import EncodedVideo
import torchshow as ts
from einops import rearrange, repeat

# export XRT_TPU_CONFIG="localservice;0;localhost:51011"
# export PT_XLA_DEBUG=1
# export USE_TORCH=ON
# export XLA_USE_BF16=1
# %load_ext autoreload
# %autoreload 2

sys.path.append(join(os.path.dirname(os.path.abspath('..')), 'hand4whole'))
sys.path.append(join(join(os.path.dirname(os.path.abspath('..')), 'hand4whole'),'main'))
sys.path.append(join(join(os.path.dirname(os.path.abspath('..')), 'hand4whole'),'common'))

from model import get_model

sys.path.append(join(os.path.abspath('..'), 'datasets'))
sys.path.append(join(os.path.abspath('..'), 'processing'))
sys.path.append(join(os.path.abspath('..'), 'core'))
sys.path.append(join(os.path.abspath('..'), 'rendering'))

from configuration import cf
from stream_data import Streamer
from my_sampler import MyDistributedBatchSampler
from renderer import video_masking, write_output
from model_net import PerceiverIO
from miscellaneous import batch_dictionary, batch_dictionary_to_list
from vectorize import input_to_tensor, fourier_encode

class FLAGS:

    # Training parameters

    num_epochs = 6
    batch_size = 1
    learning_rate = 0.001
    num_workers = 2
    num_cores = 8
    print_step = 5

    # Simulation parameters

    acting_ratio = 0.6
    duration_total=2
    duration = 1
    duration_future = 0.25

def my_collate(batch):
    path_ids = [item[0] for item in batch]
    stream_nbs = [item[1] for item in batch]
    inputs = [item[2] for item in batch]
    targets = [item[3] for item in batch]
    return (path_ids, stream_nbs, inputs, targets)

def create_dataloader( flags):

    num_workers = flags.num_workers
    batch_size = flags.batch_size
    duration = flags.duration
    duration_total = flags.duration_total
    duration_future = flags.duration_future

    # Dataloader

    dataset = Streamer(join(os.path.dirname(os.getcwd()),'data_test'), duration, duration_future, duration_total,batch_size)

    sampler = None
    if (xm.xrt_world_size() > 1):
        sampler = MyDistributedBatchSampler (#torch.utils.data.distributed.DistributedSampler( 
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=my_collate, drop_last=True, sampler=sampler)
    
    return train_loader


def create_model(device, flags):

    # Model parameters

    len_video_frame = cf.len_video_frame
    len_audio_frame = cf.len_audio_frame
    query_dim = cf.query_dim
    video_latent_dim = cf.video_latent_dim
    audio_latent_dim = cf.audio_latent_dim

    nb_frames = flags.duration*cf.fps
    frames_future = int(flags.duration_future*cf.fps)

    len_vid = nb_frames*len_video_frame
    len_audio = nb_frames*len_audio_frame
    video_future = frames_future*len_video_frame
    audio_future = frames_future*len_audio_frame

    special_compression = [i for i in range(video_future)] + [i for i in range(len_vid, len_vid+audio_future)]
    for k in range(len(cf.body_parts)):
        special_compression += [ len_vid + len_audio + k*nb_frames + i for i in range(frames_future) ]

    # Projection to return modalities, enhancement

    modality_parameters = {}
    modality_parameters['audio'] = audio_latent_dim
    modality_parameters['video'] = video_latent_dim
    for part, params in cf.body_parts.items():
        modality_parameters[part] = query_dim - 2*128 - params

    projection_parameters = dict(cf.body_parts)
    projection_parameters.update({'audio': 16})

    # Instantiate model
        
    model = PerceiverIO(
            2,
            query_dim,
            special_compression = special_compression,
            device=device,
            projection_parameters=projection_parameters,
            modality_parameters=modality_parameters,
            max_batch_size = 1
        );
    model = model.to(device)

    model.modality['audio'] = nn.Parameter(torch.randn(audio_latent_dim, device=device))
    model.modality['video'] = nn.Parameter(torch.randn(video_latent_dim, device=device))
    for part, params in cf.body_parts.items():
        model.modality[part] = nn.Parameter(torch.randn(query_dim - 2*128 - params, device=device))

    # Queries

    queries = []
    audio_query = repeat( torch.cat([model.modality['audio'], torch.zeros(16, device=device)]), '... -> n ...', n=audio_future)
    audio_pos = fourier_encode(torch.linspace(-1, 1, audio_future), audio_future/2, num_bands=128).to(device)
    queries.append(torch.cat([audio_query, audio_pos], dim=-1).to(device))
    for part, params in cf.body_parts.items():
        part_query = repeat( torch.cat([model.modality[part], torch.zeros(params, device=device)]), '... -> n ...', n=frames_future)
        part_pos = fourier_encode(torch.linspace(-1, 1, frames_future), frames_future/2, num_bands=128).to(device)
        queries.append(torch.cat([part_query, part_pos], dim=-1).to(device))
    queries=torch.cat(queries)

    return model, queries
    
# from jax_smi import initialise_tracking
# initialise_tracking()

def _mp_fn(index, flags):

    # Parameters

    batch_size = flags.batch_size
    learning_rate = flags.learning_rate

    duration = flags.duration
    duration_total = flags.duration_total
    duration_future = flags.duration_future
    acting_ratio = flags.acting_ratio

    len_audio_frame = cf.len_audio_frame
    nb_frames = flags.duration*cf.fps
    frames_future = int(flags.duration_future*cf.fps)
    len_audio = nb_frames*len_audio_frame
    len_audio_raw = len_audio*cf.sample_per_frame
    audio_future = frames_future*len_audio_frame
    max_stream_nbs = (duration_total-duration)/duration_future

    # Instantiate model, dataloader and optimizer

    deviceEval = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = xm.xla_device()
    train_loader = create_dataloader(flags)
    mp_device_loader = pl.MpDeviceLoader(train_loader, device)
    model, queries = create_model(device, flags)
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []

    # smpl-x snapshot load only once
    
    model_path = '../../hand4whole/demo/snapshot_6.pth.tar'
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model_smplx = get_model('test')
    model_smplx = DataParallel(model_smplx)
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    model_smplx.load_state_dict(ckpt['network'], strict=False)
    model_smplx = model_smplx.eval().to(device)
    
    for epoch in tqdm(range(flags.num_epochs), total=flags.num_epochs):
        target_display = {part: torch.tensor([], device=device) for part in ['video', 'audio']+list(cf.body_parts.keys())}
        pred_display = {part: torch.tensor([], device=device) for part in ['audio']+list(cf.body_parts.keys())}
        train_loss = []
        last_path_id = None
        pose_shapes = None
        len_loader = int(len(train_loader)/batch_size)
        for i, (path_ids, stream_nbs, inputs, targets) in tqdm(enumerate(mp_device_loader), total=len_loader):
            if(xm.get_ordinal()==1):
                print("tpu:",xm.get_ordinal(), 'iter:', i, "id:", path_ids, "stream:", stream_nbs)


            # Clear memory for new sequence

            if( path_ids[0] != last_path_id):
                last_path_id = path_ids[0]
                model.reset_memories()
                audio_mem = torch.tensor([], device=device)
                body_mem = {part: torch.tensor([], device=device) for part in list(cf.body_parts.keys())}
            pose_shapes = [torch.mean(torch.tensor(input['smplx_shape']), dim=0) for input in inputs]
            acting =  (stream_nbs[0]+1)/max_stream_nbs >= acting_ratio
        
            # append previous predictions and mask

            if( len(audio_mem) > 0  and acting):
                _, len_audio_mem = audio_mem.shape
                _, len_body_mem, _ = list(body_mem.values())[0].shape
                for i in range(len(inputs)):
                    audio_mem_filled = torch.cat([inputs[i]["audio"][:-len_audio_mem].to(device), audio_mem[i]])
                    inputs[i]["audio"] =  ( inputs[i]["audio"] + audio_mem_filled )/2
                    for part in cf.body_parts.keys():
                        torch.cat([torch.tensor(inputs[i][part], device=device)[:-len_body_mem, :], body_mem[part][i]])
                    inputs_i_vect = {part: item.to(device) if(torch.is_tensor(item)) else torch.tensor(item).to(device) for part, item in inputs[i].items()}
                    inputs[i]["video"] = video_masking(inputs[i]["video"]*255, inputs_i_vect, model_smplx)/255
                
        
            # remember target 

            if(epoch % FLAGS.print_step == 0 and path_ids[0] == 0 ):
                target_display['video'] = torch.cat([target_display['video'], targets[0]['video']], dim=1)
                target_display['audio'] = torch.cat([target_display['audio'], targets[0]['audio']], dim=0)
                for part in cf.body_parts.keys():
                    target_display[part] = torch.cat([target_display[part], torch.tensor(targets[0][part], device=device)], dim=0)

            # vectorize

            targets_audio = torch.tensor([target['clean_audio'] for target in targets]).to(device)
            [ target.pop(key) for key in ['video','audio','clean_audio','smplx_shape'] for target in targets ]
            targets_body  = batch_dictionary(targets)
            targets_body = {part: torch.tensor(item).to(device) for part, item in targets_body.items()}
            [ input.pop(key) for key in ['clean_audio','smplx_shape'] for input in inputs ]
            inputs = batch_dictionary(inputs)
            inputs['video'] = torch.stack(inputs['video'])
            inputs['audio'] = torch.stack(inputs['audio'])
            inputs = {part: item.to(device) if(torch.is_tensor(item)) else torch.tensor(item).to(device) for part, item in inputs.items()}
            batch = input_to_tensor(inputs, model.modality, cf.body_parts).to(device)

            # zero the parameter gradients

            optimizer.zero_grad()

            # forward + backprop + loss b

            x, aux_loss = model(batch, queries = queries)
            loss = aux_loss
            output = {}
            if(acting):
                loss_flow = 1
            else:
                loss_flow = 0
            output['audio'] = rearrange(model.projections['audio'](x[:, :audio_future, :]) ,  "b t t1 -> b (t t1)")
            start_modality = audio_future
            loss = loss + loss_flow*F.l1_loss(output['audio'], targets_audio)
            for part, params in cf.body_parts.items():
                output[part] = model.projections[part](x[:, start_modality:start_modality+frames_future, :])
                start_modality += frames_future
                loss = loss + loss_flow*F.l1_loss(output[part], targets_body[part])
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)

            # remember pred 

            if(epoch % FLAGS.print_step == 0 and path_ids[0] == 0 ):
                for part in pred_display.keys():
                    pred_display[part] = torch.cat([pred_display[part], torch.tensor(output[part][0])], dim=0)

            # update params and memory for autoregression

            if(acting):
                audio_mem = torch.cat([audio_mem, output['audio']], dim=1)
                audio_mem = audio_mem[:, max(-len(audio_mem), -len_audio_raw):].clone().detach()
                for part in cf.body_parts.keys():
                    body_mem[part] = torch.cat([body_mem[part], output[part]], dim=1)
                    body_mem[part] = body_mem[part][:, max(-len(body_mem[part]), -nb_frames):, :].clone().detach()

            # loss

            train_loss.append(loss.to(deviceEval).detach().item())
            
        # print out
        xm.master_print('##########')
        xm.master_print('epoch:',epoch)
        xm.master_print('loss:',np.mean(train_loss))
        xm.master_print('##########')

        if( epoch % FLAGS.print_step == 0 ):
            xm.save(model.state_dict(), 'weights.pt')
            xm.master_print(torch_xla.debug.metrics.metrics_report())
            if xm.is_master_ordinal():
                write_output(target_display['video'], target_display['audio'], target_display, pose_shapes[0], epoch, device, model_smplx, 'target')
                write_output(target_display['video'], pred_display['audio'], pred_display, pose_shapes[0], epoch, device, model_smplx, 'pred')


    # Close processes
    xm.rendezvous('ff')
    sys.exit(21)

if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)