import pickle
import pickle5
import pandas as pd
import cv2
import torch
from tqdm import tqdm
import os
import yaml
import numpy as np

def rot(df, image, k=1):
    '''
    Rotates image and coordinates counter-clockwise by k * 90° within image origin
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :param k: Number of times to rotate by 90°
    :return: Rotated Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0= image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    for i in range(k):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0= image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image


def fliplr(df, image):
    '''
    Flip image and coordinates horizontally
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :return: Flipped Dataframe and image
    '''
    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0= image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    R = np.array([[-1, 0], [0, 1]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    image = cv2.flip(image, 1)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0= image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image

def crop_and_pad(norm_pos, global_map, local_size, resize_factor):
    # print(traj.size(), segmentation_map.size(), semantic_map.size(), traj[0,:2])
    c, h, w = global_map.size()
    # print(c,h,w)
    center_x, center_y = norm_pos
    local_map = torch.zeros(c+1,local_size,local_size)

    if local_size == 90*3:
        half_margin = float(90*3-1)/2
        norm_center_y = int(center_y/resize_factor)
        norm_center_x = int(center_x/resize_factor)
        top    = int(norm_center_y - half_margin)
        bottom = top + local_size - 1
        # bottom = int(norm_center_y + half_margin)
        left   = int(norm_center_x - half_margin)
        right  = left + local_size - 1
        # right  = int(norm_center_x + half_margin)

    else:
        half_margin = float(local_size-1) / 2
        
        top    = int(center_y/resize_factor - half_margin)
        bottom = int(center_y/resize_factor + half_margin)
        left   = int(center_x/resize_factor - half_margin)
        right  = int(center_x/resize_factor + half_margin)

    # print(local_size,top, bottom, left, right)

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


def preprocess_image_for_segmentation(images, classes=6):   
    for key, im in images.items():
        im = [(im == v) for v in range(classes)]
        im = np.stack(im, axis=-1)

        im = im.transpose(2, 0, 1).astype('float32')
        im = torch.Tensor(im)
        images[key] = im

        
def resize(images, factor, seg_mask=True):
    for key, image in images.items():
        if seg_mask:
            images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        else:
            images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

def pad(images, division_factor=32):
    """ Pad image so that it can be divided by division_factor, as many architectures such as UNet needs a specific size
    at it's bottlenet layer"""
    for key, im in images.items():
        if im.ndim == 3:
            H, W, C = im.shape
        else:
            H, W = im.shape
        H_new = int(np.ceil(H / division_factor) * division_factor)
        W_new = int(np.ceil(W / division_factor) * division_factor)
        im = cv2.copyMakeBorder(im, 0, H_new - H, 0, W_new - W, cv2.BORDER_CONSTANT)
        images[key] = im