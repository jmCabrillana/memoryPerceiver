from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import sys
import time
import pickle
from more_itertools import grouper
from IPython.display import HTML

import torch
import torch.multiprocessing as mp
from torchaudio.transforms import Resample
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.utils import draw_bounding_boxes
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
from speechbrain.pretrained import SepformerSeparation as separator
import torchshow as ts
from einops import rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
deviceEval = device


sys.path.append(join(os.path.dirname(os.path.abspath('..')), 'hand4whole'))
sys.path.append(join(join(os.path.dirname(os.path.abspath('..')), 'hand4whole'),'main'))
sys.path.append(join(join(os.path.dirname(os.path.abspath('..')), 'hand4whole'),'common'))

from config import cfg
from model import get_model
# from utils.preprocessing import process_bbox, generate_patch_image
# from utils.human_models import smpl_x
# from utils.vis import render_mesh, save_obj

sys.path.append(join(os.path.abspath('..'), 'datasets'))
sys.path.append(join(os.path.abspath('..'), 'processing'))
sys.path.append(join(os.path.abspath('..'), 'core'))
sys.path.append(join(os.path.abspath('..'), 'rendering'))

from configuration import cf
from box_detection import interpolate_bbox, Resize_with_pad
from videoTransform import transform_video, transform_audio
# from renderer import render_silhouettes, batch_intrinsic_camera



# read write data 

def write_seq(dict, dst):
    len_seq = len(dict['video'])
    for i in range(len_seq):
        single_seq = []
        single_seq.append(dict['video'])
        single_seq.append(dict['audio'][i*cf.sample_per_frame : (i+1)*cf.sample_per_frame])
        single_seq.append(dict['clean_audio'][i*cf.sample_per_frame : (i+1)*cf.sample_per_frame])
        for part in cf.body_parts.keys():
            single_seq.append(cf.body_parts[part])
        pickle.dump(single_seq, dst, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(pickle.dumps(single_seq)))
    dst.flush()

def read_seq(path, start, stop):
    start_frame = int(start * cf.fps)
    end_frame = int(stop * cf.fps)
    seq_file = open(path, 'rb')
    seq_file.seek(start_frame*cf.seq_byte_size, os.SEEK_CUR)
    seq_dict = {}
    seq_dict['video'] = []
    seq_dict['audio'] = []
    for part in cf.body_parts.keys():
        seq_dict[part] = []

    # Read pickle file 

    for i in range(start_frame, end_frame+1):
        seq_array = pickle.load(seq_file)
        seq_dict['video'].append(seq_array[0])
        seq_dict['audio'] += seq_array[1]
        seq_dict['clean_audio'] += seq_array[1]
        part_index = 3
        for part in cf.body_parts.keys():
            seq_dict[part].append(seq_array[part_index])
            part_index+=1

    seq_file.close()

    return seq_dict

def body_from_video(video, model_bbox, model_smplx, draw_bbox=False):

    deviceEval = next(model_bbox.parameters()).device

    # compute bbox

    video = rearrange(video ,  "c t h w -> t c h w")/255
    _, _, h, w = video.shape
    frames=len(video)
    with torch.no_grad():
        res=model_bbox(video.to(deviceEval))
    bbox=interpolate_bbox(res, h, w).int()

    # make patches

    vids=[]
    max_w = max( bbox[:, 2]-bbox[:, 0]+1 )
    max_h = max( bbox[:, 3]-bbox[:, 1]+1 )
    resize_pad = Resize_with_pad(max_w, max_h)
    for i in range(frames):
        xmin, ymin, xmax, ymax = bbox[i]
        vid=video[i][:, ymin:ymax, xmin:xmax ]
        dw, dh, vid = resize_pad(vid)
        bbox[i] = torch.tensor([xmin-dw, ymin-dh, xmax+dw, ymax+dh])
        vids.append(vid)
    vids=torch.stack(vids)
    if(draw_bbox):
        ts.show(draw_bounding_boxes((video*255)[0].byte(), bbox.int()))

    # compute targets

    inputs, targets, meta_info = {'img': vids.to(deviceEval)}, {}, {}
    with torch.no_grad():
        out = model_smplx(inputs, targets, meta_info, 'test')
    bbox = bbox.float()
    bbox[:, [0,2]] = bbox[:, [0,2]]/w
    bbox[:, [1,3]] = bbox[:, [1,3]]/h
    out['bbox']=bbox.to(deviceEval)

    del out['img']
    del out['joint_img']
    del out['smplx_mesh_cam']

    return out

def pickle_media_pose(path, model_bbox, model_smplx, model_sepformer, duration = 1):

    deviceEval = next(model_bbox.parameters()).device
    # subsampler = Resample(cf.sample_rate, cf.sepformer_rate).to(deviceEval)
    # upsampler = Resample(cf.sepformer_rate, cf.sample_rate).to(deviceEval)

    stream = EncodedVideo.from_path(path)
    duration = stream.duration
    slice_duration = 1
    nb_frames = slice_duration * cf.fps
    seq_path = os.path.dirname(path) + '/seq/' + os.path.basename(path)[:-4] + '.bin'
    seq_file = open(seq_path, 'wb')

    for i in range(0, 1, slice_duration): #int(duration)):

        t0 = time.time()

        input = stream.get_clip(start_sec=i, end_sec = min(i+1, duration))

        print('get clip: ', time.time()-t0)
        t0 = time.time()

        # Extract body

        input.update(body_from_video(UniformTemporalSubsample(nb_frames)(input['video']), model_bbox, model_smplx))

        print('body: ', time.time()-t0)
        t0 = time.time()

        # Transform media, add clean sound

        input = transform_audio(nb_frames)(transform_video(nb_frames)(input)) 

        print('transform media: ', time.time()-t0)
        t0 = time.time()

        model_sepformer; #(subsampler(input["audio"].to(deviceEval)).unsqueeze(0))
        # input['clean_audio'] = upsampler(model_sepformer(subsampler(input["audio"].to(deviceEval)).unsqueeze(0)).squeeze(-1)).squeeze(-1).to(deviceEval)
        input['clean_audio'] = input['audio']

        print('clean audio: ', time.time()-t0)
        t0 = time.time()

        # Flush to free some memory
        for key in input.keys():
            input[key] = input[key].tolist()
        write_seq(input, seq_file)
        print('write mem: ', time.time()-t0)

    seq_file.close()

def pickle_media_pose_group(pathes, model_bbox, model_smplx, model_sepformer ):
    for path in pathes:
        if(path is not None):
            print(pathes)
            pickle_media_pose(path, model_bbox, model_smplx,  model_sepformer)
    return True

if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deviceEval = device

    # rcnn load

    model_bbox = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    model_bbox=model_bbox.eval()
    model_bbox=model_bbox.to(deviceEval)

    # smpl-x snapshot load

    model_path = '../../hand4whole/demo/snapshot_6.pth.tar'
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model_smplx = get_model('test')
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    model_smplx.load_state_dict(ckpt['network'], strict=False)
    model_smplx = model_smplx.eval().to(deviceEval)

    # sepformer load

    model_sepformer = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir='pretrained_models/sepformer-wham-enhancement', run_opts={"device":"cuda"})
    model_sepformer = model_sepformer.eval()

    # multiprocessing
    
    num_processes = 4
    root_path = '/home/jeanmanuel/memory-perceiver/memory_perceiver/data_test'
    pathes = [join(root_path, f) for f in listdir(root_path) if isfile(join(root_path, f))]
    pathes =  [group for group in grouper(int(len(pathes)/num_processes), pathes)]
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=pickle_media_pose_group, args=(pathes[rank], model_bbox, model_smplx, model_sepformer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # multi_pool = mp.Pool(processes=5)
    # predictions = multi_pool.map(pickle_media_pose_group, (pathes, model_bbox, model_smplx, model_sepformer))
    # multi_pool.close() 
    # multi_pool.join()