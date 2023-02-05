
import os
from os.path import join
import sys
import numpy as np
import torch
from torch import einsum
import torch.nn.functional as F
import torchvision
import torchaudio
from einops import rearrange, repeat

# Data structures and functions for rendering
# from pytorch3d.io import load_objs_as_meshes, load_obj
# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import (
#     RasterizationSettings, 
#     MeshRenderer, 
#     MeshRasterizer,  
#     SoftPhongShader,
#     SoftSilhouetteShader,
#     TexturesVertex,
#     PerspectiveCameras,
#     PointLights
# )

sys.path.append(join(join(os.path.abspath('..'), 'hand4whole'),'main'))
from config import cfg

sys.path.append(join(join(os.path.abspath('...'), 'hand4whole'),'common'))
from utils.human_models import smpl_x

sys.path.append(join(os.path.abspath('..'), 'main'))
from configuration import cf


def render_silhouettes(img, mesh, faces, cam_params, device, hd=False):

    frames, _, h, w = img.shape

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    # R, T = look_at_view_transform(2.7, 0, 180) 
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    R=torch.eye(3)
    R[1,1]=-1
    R[0,0]=-1
    cameras = PerspectiveCameras(device=device, focal_length=cam_params['focal'], 
                                    principal_point=cam_params['princpt'], in_ndc=False,
                                    image_size=torch.tensor([[h, w]]), R=repeat(R, 'd1 d2 -> f d1 d2', f = frames))

    if(hd):
        raster_settings = RasterizationSettings(
            image_size=(h, w), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )
    else:
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=(h, w), 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=1
        )
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )

    vert=mesh
    faces=torch.tensor(faces.astype(int)).long()
    verts_rgb = torch.ones_like(vert)
    # verts_rgb[:, 1] = 1.0 # green
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    meshes = Meshes(verts=vert.to(device), faces=faces.to(device), textures=textures)

    #render 

    images = silhouette_renderer(meshes)

    # mask images

    if(hd):
        output_images = images[ ..., 3]
    else:
        output_images = einsum('b c y x, b y x -> b c y x', img.to(device), 500*F.relu(0.002-images[ ..., 3]))
        
    return output_images

def render_projection(video, mesh, cam):
    rendered_video = video.clone()
    device = mesh.device
    c, t, h, w = rendered_video.shape
    _, m, _ = mesh.shape
    dxy = 0

    # Projection:
    # x = fx * X / Z + px
    # y = fy * Y / Z + py

    mesh_proj_X = einsum('t, t m -> t m', cam['focal'][:,0], mesh[:,:,0]/mesh[:,:,2]) + repeat(cam['princpt'][:,0], 't -> t m', m=mesh.shape[1])
    mesh_proj_Y = einsum('t, t m -> t m', cam['focal'][:,1], mesh[:,:,1]/mesh[:,:,2]) + repeat(cam['princpt'][:,1], 't -> t m', m=mesh.shape[1])
    coords = rearrange([mesh_proj_X,  mesh_proj_Y], "xy t m -> t m xy").long()

    # Set corresponding pixels to 0
    # for frame in range(t):
    #     for i in range(0, len(coords[frame]), 3):
    #         coord = coords[frame, i]
    #         for x in range(max(0, coord[0]-dxy), min(w, coord[0]+1+dxy)):
    #             for y in range(max(0, coord[1]-dxy), min(h, coord[1]+1+dxy)):
    #                 rendered_video[:,frame, y, x] = 0

    for frame in range(t):
        rendered_video[:, frame, 
                        torch.min(torch.ones(m).to(device)*(h-1), torch.max(torch.zeros(m).to(device), coords[frame,:,1])).long(),
                        torch.min(torch.ones(m).to(device)*(w-1), torch.max(torch.zeros(m).to(device), coords[frame,:,0])).long()
                    ] = 0
                    
    return rendered_video

def video_masking(video, body, model_smplx, mode='projection'):
    device = video.device
    c, frames, h, w = video.shape
    _, _, mesh = model_smplx.module.get_coord(body['smplx_root_pose'], body['smplx_body_pose'], body['smplx_lhand_pose'],  body['smplx_rhand_pose'], body['smplx_jaw_pose'],
                                             body['smplx_shape'],  body['smplx_expr'], body['cam_trans'], 'test') 
    cam_params = batch_intrinsic_camera(body['bbox'], w, h)
    if(mode=='projection'):
        rendered_video = render_projection(video, mesh, cam_params)
    else:
        vis_vid = rearrange(video, 'c t h w -> t c h w') * 255
        faces = repeat(smpl_x.face, 'd1 d2 -> f d1 d2', f = frames)
        rendered_video = render_silhouettes(vis_vid, mesh, faces, cam_params, device, hd=(mode!='silhouette'))
        rendered_video = rearrange(rendered_video/255, 't c h w -> c t h w')
    return rendered_video

def write_output(video, audio, body, pose_shapes, epoch, device, model_smplx, name):
    len_seq, _ = body['smplx_root_pose'].shape
    body.update({'smplx_shape': repeat(pose_shapes, '... -> n ...', n=len_seq).to(device)})
    body = {part: item.to(device) if(torch.is_tensor(item)) else torch.tensor(item).to(device) for part, item in body.items()}
    video_pred = video_masking(video[:,-len_seq:,:,:].to(device)*255, body, model_smplx)/255
    torchvision.io.write_video('./monitor/test_'  + str(epoch) + '_vid_' + name + '.mp4', 255*rearrange(video_pred, 'c f h w -> f h w c').cpu(), cf.fps)
    torchaudio.save('./monitor/test_'  + str(epoch) + '_audio_' + name + '.wav', audio[None, :].detach().cpu(), cf.sample_rate)

    

def intrinsic_camera(bbox, w, h):
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * (bbox[2]-bbox[0]) * w, cfg.focal[1] / cfg.input_body_shape[0] * (bbox[3]-bbox[1]) * h]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * (bbox[2]-bbox[0]) * w + bbox[0] * w, cfg.princpt[1] / cfg.input_body_shape[0] * (bbox[3]-bbox[1]) * h + bbox[1] * h]
    return {'focal': focal, 'princpt': princpt}

def batch_intrinsic_camera(bbox, w, h):
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * (bbox[:, 2]-bbox[:, 0])*w, cfg.focal[1] / cfg.input_body_shape[0] * (bbox[:, 3]-bbox[: ,1])*h]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * (bbox[:, 2]-bbox[:, 0])*w + bbox[:, 0]*w, cfg.princpt[1] / cfg.input_body_shape[0] * (bbox[:, 3]-bbox[:, 1])*h + bbox[:, 1]*h]
    focal, princpt = map(lambda list_xy: rearrange(list_xy, 'xy b -> b xy'), [focal, princpt])
    return {'focal': focal, 'princpt': princpt}

def batch_intrinsic_camera_original(frames, w, h):
    focal = torch.tensor([cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]])
    princpt = torch.tensor([cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]])
    focal, princpt = map(lambda list_xy: repeat(list_xy, 'xy -> f xy', f=frames), [focal, princpt])
    return {'focal': focal, 'princpt': princpt}
