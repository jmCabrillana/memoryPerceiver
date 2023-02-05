import torch
import torchvision.transforms.functional as vF

from einops import rearrange, repeat
class Resize_with_pad:
    def __init__(self, w=1024, h=768):
        self.w = w
        self.h = h

    def __call__(self, image):

        _, w_1, h_1 = image.shape
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if (abs(ratio_1 - ratio_f) < 0.01):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = vF.pad(image, (0, hp, 0, hp), 0, "constant")
                return 0, hp, vF.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = vF.pad(image, (wp, 0, wp, 0), 0, "constant")
                return wp, 0, vF.resize(image, [self.h, self.w])
            else:
                return 0, 0, vF.resize(image, [self.h, self.w]) # for errors
        else:
            return 0, 0, vF.resize(image, [self.h, self.w])

def interpolate_bbox(res, h, w):
    frames=len(res)
    bboxes=torch.zeros(frames, 4)
    dummy_bbox=torch.tensor([0, 0, w, h])
    first_bbox=None
    to_interpolate=[]
    for index_frame in range(frames):
        nb_bboxes=len(res[index_frame]["boxes"])
        if( nb_bboxes > 0 ):

            # take the leftmost bbox, 0 as the result is ordered

            index_leftmost = float('inf')
            for index_bbox in range(nb_bboxes):

                # label 1 is a person

                if( res[index_frame]["labels"][index_bbox].item()==1 and index_bbox<index_leftmost ):
                    index_leftmost=index_bbox

            if( index_leftmost != float('inf') ):
                bbox_leftmost=res[index_frame]["boxes"][index_leftmost]
                to_interpolate.append({"index": index_frame, "box": bbox_leftmost})
                if(first_bbox == None):
                    first_bbox=bbox_leftmost

    # no bbox found, return dummy bbox

    if(first_bbox == None):
        for i in range(frames):
            bboxes[i]=dummy_bbox
        return  bboxes

    # Interpolate where no bbox has been found, first add start and end

    if( to_interpolate[0]["index"] != 0 ):
        to_interpolate.insert(0, {"index": 0, "box": first_bbox})
    if( to_interpolate[-1]["index"] != frames-1 ):
        to_interpolate.append({"index": frames-1, "box": to_interpolate[-1]["box"]})
    for k in range(len(to_interpolate)-1):
        t0 = to_interpolate[k]["index"]
        t1 = to_interpolate[k+1]["index"]
        for i in range(t0, t1):
            bboxes[i] = to_interpolate[k]["box"]*(t1-i)/(t1-t0) + to_interpolate[k+1]["box"]*(i-t0)/(t1-t0)
    bboxes[frames-1] = to_interpolate[-1]["box"]
    return bboxes


def sanitize_bbox(bbox):
    out_bbox = torch.zeros(4)
    for i in range(len(bbox)):
        out_bbox[i] = min(1, max(0, bbox[i]))
    return out_bbox

def bbox_trans(mesh, bbox, h, w, cam_params, device):
    bbox = bbox.to(device)
    trans_x = cam_params[:, 1] +2 * ( (bbox[:, 2]+bbox[:, 0])*w/2 - w/2) / ( cam_params[:, 0] * ( (bbox[:, 2]-bbox[:, 0])*w ) )
    trans_y = cam_params[:, 2] +2 * ( (bbox[:, 3]+bbox[:, 1])*h/2 - h/2) / ( cam_params[:, 0] * ( (bbox[:, 3]-bbox[:, 1])*h ) )
    trans_z = 2 * 5000 / ( cam_params[:, 0] * ( (bbox[:, 3]-bbox[:, 1])*h ) )
    trans = rearrange([trans_x, trans_y, trans_z], 'xyz f -> f xyz')
    _, d, _ = mesh.shape
    return repeat(trans, 'f xyz -> f d xyz', d=d)