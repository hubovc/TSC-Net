import torch
import math
import time
import random
import numpy as np

def euclidean_distance(traj_0, traj_1):
    (nb,nd,np,nf) = traj_0.size()

    dist = (traj_0 - traj_1) ** 2
    dist = dist.sum(dim=1).sqrt().reshape(nb,1,np,nf)

    return dist

def xy_to_polar(p0, p1):
    # p0, p1 shape: B x 2 x N x F
    pi = torch.acos(torch.zeros(1)).item() * 2 

    nb, nd, np, nf = p0.size()
    p_diff = p1 - p0
    dist = torch.sqrt( p_diff[:,0:1,:,:].pow(2) + p_diff[:,1:,:,:].pow(2) )
    angle = torch.asin(p_diff[:,1:,:,:] / dist)

    angle[p_diff[:,0:1,:,:]<=0] = angle[p_diff[:,0:1,:,:]<=0] * (-1) + pi
    angle[(p_diff[:,0:1,:,:]>0)*(p_diff[:,1:,:,:]<0)] = angle[(p_diff[:,0:1,:,:]>0)*(p_diff[:,1:,:,:]<0)] + pi * 2

    angle[torch.isnan(angle)] = -1
    angle[torch.isinf(angle)] = -1

    pos_polar = torch.cat([dist, angle],1)

    return pos_polar

def positional_encoding_2d_center(D, W, H, scale ):
    pe = torch.zeros(1,D,H,W).cuda()

    for y in range(-int(H/2),int(H/2),scale[0]):
        ty = torch.tensor(y).float()
        for x in range(-int(W/2),int(W/2),scale[1]):
            tx = torch.tensor(x).float()
            for d in range(int(D/4)):
                td = torch.tensor(d).float()
                pe[0,int(2*d),y,x]            = torch.sin( tx / torch.pow( 10000, 4 * td / D ) )
                pe[0,int(2*d+1),y,x]          = torch.cos( tx / torch.pow( 10000, 4 * td / D ) )
                pe[0,int(2*d+int(D/2)),y,x]   = torch.sin( ty / torch.pow( 10000, 4 * td / D ) )
                pe[0,int(2*d+int(D/2))+1,y,x] = torch.cos( ty / torch.pow( 10000, 4 * td / D ) )
    
    return pe

def positional_encoding_1d(D, T, scale=1):
    pe = torch.zeros(1,D,1,T).cuda()

    for t in range(T,scale):
        tt = torch.tensor(t).float()
        for d in range(int(D/2)):
            td = torch.tensor(d).float()
            pe[0,int(2*d),y,x]   = torch.sin( tt / torch.pow( 10000, 2 * td / D ) )
            pe[0,int(2*d+1),y,x] = torch.cos( tt / torch.pow( 10000, 2 * td / D ) )
    
    return pe

def traj_to_polar_traj(traj, obs_pos=7):
    # traj shape: N x 20 x 2
    pi = torch.acos(torch.zeros(1)).item() * 2 

    np, nf, nd = traj.size()
    traj_diff = traj[:,1:] - traj[:,:-1]
    dist = torch.sqrt( traj_diff[:,:,0].pow(2) + traj_diff[:,:,1].pow(2) )
    angle = torch.asin(traj_diff[:,:,1] / dist)

    angle[traj_diff[:,:,0]<=0] = angle[traj_diff[:,:,0]<=0] * (-1) + pi
    angle[(traj_diff[:,:,0]>0) * (traj_diff[:,:,1]<0)] = angle[(traj_diff[:,:,0]>0) * (traj_diff[:,:,1]<0)] + pi * 2
    
    angle[torch.isnan(angle)] = -1
    angle[torch.isinf(angle)] = -1

    traj_polar = torch.stack([dist, angle],2)
    traj_polar = torch.cat((traj_polar[:,:1,:],traj_polar),1)

    return traj_polar

def crop_and_pad(norm_pos, global_map, local_size, resize_factor):
    # print(traj.size(), segmentation_map.size(), semantic_map.size(), traj[0,:2])
    c, h, w = global_map.size()
    center_x, center_y = norm_pos
    local_map = torch.zeros(c+1,local_size,local_size)

    if local_size == 45:
        half_margin = 22
        norm_center_y = int(center_y/resize_factor)
        norm_center_x = int(center_x/resize_factor)
        top    = norm_center_y - half_margin
        bottom = norm_center_y + half_margin
        left   = norm_center_x - half_margin
        right  = norm_center_x + half_margin
    else:
        half_margin = float(local_size-1) / 2
        
        top    = int(center_y/resize_factor - half_margin)
        bottom = int(center_y/resize_factor + half_margin)
        left   = int(center_x/resize_factor - half_margin)
        right  = int(center_x/resize_factor + half_margin)

    true_top    = min( max(0, top),    h-1 )
    true_bottom = min( max(0, bottom), h-1 )
    true_left   = min( max(0, left),   w-1 )
    true_right  = min( max(0, right),  w-1 )

    # true_top    = max(0, top)
    # true_bottom = min(h-1, bottom)
    # true_left   = max(0, left)
    # true_right  = min(w-1, right)
    true_h      = true_bottom-true_top+1 if true_bottom != true_top else 0
    true_w      = true_right-true_left+1 if true_left != true_right else 0

    pad_top    = true_top - top
    # pad_bottom = bottom - true_bottom
    pad_left   = true_left - left
    # pad_right  = right - true_right

    crop_map = global_map[:,true_top:true_top+true_h,true_left:true_left+true_w]
    local_map[:c,pad_top:pad_top+true_h,pad_left:pad_left+true_w] = crop_map

    pad_channel = torch.zeros(local_size,local_size)
    pad_channel[local_map.sum(0)==0] = 1

    local_map[c] = pad_channel

    return local_map