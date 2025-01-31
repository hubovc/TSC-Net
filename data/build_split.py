import pickle
import pickle5
import pandas as pd
import cv2
import torch
from tqdm import tqdm
import os
from easydict import EasyDict
import yaml

from utils_preprocessing import *

with open('../configs/two_stage_ssd.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = EasyDict(config)

resize_factor = config.resize_factor

ori_global_size = config.ori_global_size
global_size = config.global_size
global_anchor_size = config.global_anchor_size
global_step = config.global_step

ori_local_size = config.ori_local_size
local_size = config.local_size
local_anchor_size = config.local_anchor_size
local_step = config.local_step

obs = config.obs_length
traj_len = config.clip_length

global_steps = torch.FloatTensor(list(range(int(global_step/2),ori_global_size,global_step)))
global_anchor_coord_xy = torch.zeros(1,2,global_anchor_size,global_anchor_size)
global_anchor_coord_xy[:,0] = global_steps.reshape(1,1,1,global_anchor_size).repeat(1,1,global_anchor_size,1)
global_anchor_coord_xy[:,1] = global_steps.reshape(1,1,global_anchor_size,1).repeat(1,1,1,global_anchor_size)
ori_global_pos = torch.FloatTensor([ori_global_size/2,ori_global_size/2]).reshape(1,2,1,1).repeat(1,1,global_anchor_size,global_anchor_size)
global_anchor_coord_xy = global_anchor_coord_xy - ori_global_pos

local_steps = torch.FloatTensor(list(range(int(local_step/2),ori_local_size,local_step)))
local_anchor_coord_xy = torch.zeros(1,2,local_anchor_size,local_anchor_size)
local_anchor_coord_xy[:,0] = local_steps.reshape(1,1,1,local_anchor_size).repeat(1,1,local_anchor_size,1)
local_anchor_coord_xy[:,1] = local_steps.reshape(1,1,local_anchor_size,1).repeat(1,1,1,local_anchor_size)
ori_local_pos = torch.FloatTensor([ori_local_size/2,ori_local_size/2]).reshape(1,2,1,1).repeat(1,1,local_anchor_size,local_anchor_size)
local_anchor_coord_xy = local_anchor_coord_xy - ori_local_pos

def augment_data(data, image_path='raw/scene/', image_file='_mask.png', seg_mask=True):
    '''
    Perform data augmentation
    :param data: Pandas df, needs x,y,metaId,sceneId columns
    :param image_path: example - 'data/SDD/val'
    :param images: dict with key being sceneId, value being PIL image
    :param image_file: str, image file name
    :param seg_mask: whether it's a segmentation mask or an image file
    :return:
    '''
    images = {}
    images_ori = {}

    ks = [1, 2, 3]
    for scene in data.sceneId.unique():
        im_path = image_path + scene + image_file
        # im_path = os.path.join(image_path, scene, image_file)

        if seg_mask:
            im = cv2.imread(im_path, 0)
        else:
            im = cv2.imread(im_path)
        images[scene] = im
        images_ori[scene] = im
    data_ = data.copy()  # data without rotation, used so rotated data can be appended to original df
    k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
    for k in ks:
        metaId_max = data['metaId'].max()
        for scene in data_.sceneId.unique():
            im_path = image_path + scene + image_file

            if seg_mask:
                im = cv2.imread(im_path, 0)
            else:
                im = cv2.imread(im_path)

            data_rot, im = rot(data_[data_.sceneId == scene], im, k)
            # image
            rot_angle = k2rot[k]
            images[scene + rot_angle] = im

            data_rot['sceneId'] = scene + rot_angle
            data_rot['metaId'] = data_rot['metaId'] + metaId_max + 1
            data = data.append(data_rot)

    metaId_max = data['metaId'].max()
    for scene in data.sceneId.unique():
        im = images[scene]
        data_flip, im_flip = fliplr(data[data.sceneId == scene], im)
        data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
        data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
        data = data.append(data_flip)
        images[scene + '_fliplr'] = im_flip

    return data, images, images_ori


def build_global_cls_gt_mask_xy(scene_sample):  
    np = scene_sample['num_ped']
    cell_map_xy_goal_gt = torch.zeros(np,3,global_anchor_size,global_anchor_size)
    cell_map_xy_goal_mask = torch.zeros(np,3,global_anchor_size,global_anchor_size)
    cell_map_xy_goal_mask[:,0] = 1

    traj_xy_raw = scene_sample['traj_raw_xyt'][:,:2,:,:]

    traj_xy_norm_pos = traj_xy_raw[:,:,:,obs-1:obs]
    traj_xy_norm = traj_xy_raw - traj_xy_norm_pos.repeat(1,1,1,traj_len)
    traj_xy_norm_goal = traj_xy_norm[:,:,:,-1:]
    # traj_xy_center_goal = traj_xy_norm_goal

    cell_pos_xy_goal = torch.cat( ( ((traj_xy_norm_goal[0,0:1,:,0]+ori_global_size/2)/global_step).int(), ((traj_xy_norm_goal[0,1:,:,0]+ori_global_size/2)/global_step).int() ), 0)
    
    for p in range(np):
        ind_x, ind_y = cell_pos_xy_goal[:,p]
        # print(ind_x.item(), ind_y.item(), traj_xy_norm_goal[:,:,p,:].reshape(-1).tolist(), ori_global_size/2, global_step) 
        # print(global_anchor_coord_xy)
        # tmp = input()
        if ind_x < 0:
             ind_x = 0
        if ind_x >= global_anchor_size:
             ind_x = global_anchor_size - 1
        if ind_y < 0:
             ind_y = 0
        if ind_y >= global_anchor_size:
             ind_y = global_anchor_size - 1

        # print(ind_x, ind_y)
        min_ind_x, max_ind_x = max(0,ind_x-1), min(global_anchor_size-1,ind_x+1)
        min_ind_y, max_ind_y = max(0,ind_y-1), min(global_anchor_size-1,ind_y+1)
        # cell_map_xy_goal_mask[p,1:,min_ind_y:max_ind_y+1,min_ind_x:max_ind_x+1] = 1
        cell_map_xy_goal_mask[p,1:,ind_y,ind_x] = 1

        cell_map_xy_goal_gt[p,0,ind_y,ind_x] = 1
        cell_map_xy_goal_mask[p,0,ind_y,ind_x] = global_anchor_size
        cell_map_xy_goal_gt[p,1:] = traj_xy_norm_goal[0,:,p,:].reshape(2,1,1).repeat(1,global_anchor_size,global_anchor_size) - global_anchor_coord_xy
       
        # print(cell_map_xy_goal_mask[p,1:,:,:])
        # tmp = input()
        # print(cell_map_xy_goal_gt[p,1:,:,:]*cell_map_xy_goal_mask[p,1:,:,:])
        # print(cell_map_xy_goal_gt[p,0,:,:])
        # tmp = input()
        
    # tmp = input()

    global_anchor_coord_xyt = torch.cat((global_anchor_coord_xy,torch.ones(1,1,global_anchor_size,global_anchor_size)*12),1)

    return cell_map_xy_goal_gt, cell_map_xy_goal_mask, global_anchor_coord_xyt

def build_local_cls_gt_mask_xy(scene_sample):
    np = scene_sample['num_ped']
    cell_map_xy_pred_gt = torch.zeros(np,traj_len,3,local_anchor_size,local_anchor_size)
    cell_map_xy_pred_mask = torch.zeros(np,traj_len,3,local_anchor_size,local_anchor_size)
    cell_map_xy_pred_mask[:,:,0] = 1

    traj_xy_step = scene_sample['traj_raw_vel']
    cell_pos_xy_step = torch.cat( ( ((traj_xy_step[0,0:1,:,:]+ori_local_size/2)/local_step).int(), ((traj_xy_step[0,1:,:,:]+ori_local_size/2)/local_step).int() ), 0)
    for p in range(np):
        for f in range(traj_len):
            ind_x, ind_y = cell_pos_xy_step[:,p,f]
            if ind_x < 0:
                 ind_x = 0
            if ind_x >= local_anchor_size:
                 ind_x = local_anchor_size - 1
            if ind_y < 0:
                 ind_y = 0
            if ind_y >= local_anchor_size:
                 ind_y = local_anchor_size - 1

            min_ind_x, max_ind_x = max(0,ind_x-1), min(local_anchor_size-1,ind_x+1)
            min_ind_y, max_ind_y = max(0,ind_y-1), min(local_anchor_size-1,ind_y+1)
            cell_map_xy_pred_mask[p,f,1:,min_ind_y:max_ind_y+1,min_ind_x:max_ind_x+1] = 1

            cell_map_xy_pred_gt[p,f,0,ind_y,ind_x] = 1
            # cell_map_xy_pred_mask[p,f,0,ind_y,ind_x] = 3
            cell_map_xy_pred_gt[p,f,1:] = traj_xy_step[0,:,p,f].reshape(2,1,1).repeat(1,local_anchor_size,local_anchor_size) - local_anchor_coord_xy
            cell_map_xy_pred_gt[cell_map_xy_pred_mask==0] = 0

    local_anchor_coord_xyt = torch.cat((local_anchor_coord_xy,torch.ones(1,1,3,3)),1)

    return cell_map_xy_pred_gt, cell_map_xy_pred_mask, local_anchor_coord_xyt

def build_center_scenes(scene_sample):
    start_frame = scene_sample['start_frame']
    scene_map = scene_sample['scene_map']      
    sceneId = scene_sample['sceneId']       
    
    new_file_name = 'center_scenes/' + '%s_%05d.pkl'%(sceneId,start_frame)
    
    traj_raw_xyt = scene_sample['traj_raw_xyt']
    # print(traj_raw_xyt.size())
    # build goal center scene        
    scene_goal_list = []
    goal_center_pos = traj_raw_xyt[:,:2,:,obs-1:obs]

    num_ped = traj_raw_xyt.size(2)
    for p in range(num_ped):
        one_scene_goal_raw = crop_and_pad(goal_center_pos[0,:,p,0], scene_map, global_size, resize_factor)
        scene_goal_list.append(one_scene_goal_raw)
    scene_goal_raw = torch.stack(scene_goal_list, dim=0)
    # print(num_ped, scene_goal_raw.size())
    

    # build step center scene
    scene_step_list = []
    for p in range(num_ped):
        scene_step_p_list = []
        for f in range(traj_raw_xyt.size(3)):
            step_center_pos = traj_raw_xyt[0,:2,p,f]
            one_scene_step_raw = crop_and_pad(step_center_pos, scene_map, local_size*3, resize_factor)
            scene_step_p_list.append(one_scene_step_raw)
        scene_step_list.append(torch.stack(scene_step_p_list,0)) # T x 15 x 15 x 7
    scene_step_raw = torch.stack(scene_step_list,0)
    # print(num_ped, scene_step_raw.size())
    # tmp = input()

    one_scene_dict = {}
    one_scene_dict['scene_goal_raw'] = scene_goal_raw
    one_scene_dict['scene_step_raw'] = scene_step_raw

    # print(num_ped, scene_goal_raw.size(), scene_step_raw.size())
    # tmp = input()

    with open(new_file_name, 'wb') as f:
        pickle.dump(one_scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)





def build_traj_scene_cls_split(data_raw, scene_img_dict):
    traj_scene_cls_list = {}

    # with open('scene_seg_images.pkl','rb') as f:
    #     scene_seg_images = pickle.load(f)

    for i in range(0,len(data_raw),traj_len):
        frame_idx_list = data_raw[i:i+traj_len]['frame'].to_numpy().astype('float32')
        traj_xy = data_raw[i:i+traj_len][['x','y']].to_numpy().astype('float32')
        traj_xy = torch.FloatTensor(traj_xy)
        start_frame = int(frame_idx_list[0])
        sceneId = data_raw.iloc[i]['sceneId']

        traj_xy = torch.FloatTensor(traj_xy)

        traj_step = traj_xy[obs:] - traj_xy[obs-1:-1]
        one_step_x = traj_step[:,0].abs().max().item()
        one_step_y = traj_step[:,1].abs().max().item()

        if one_step_y > 90 or one_step_x > 90:
            continue

        scene_traj_key = sceneId + '_%06d'%start_frame
        if scene_traj_key not in traj_scene_cls_list.keys():
            traj_scene_cls_list[scene_traj_key] = {'traj_raw_xyt':[], 'frame_idx_list':frame_idx_list, 'start_frame':start_frame, 'sceneId':sceneId}
            # one_hot_img = transfer_scene_map(scene_img_dict[sceneId], resize_factor)

            traj_scene_cls_list[scene_traj_key]['scene_map'] = scene_img_dict[sceneId]
            # traj_scene_cls_list[scene_traj_key]['scene_binary_map'] = load_scene_map(sceneId)

        traj_scene_cls_list[scene_traj_key]['traj_raw_xyt'].append(traj_xy)

    traj_scene_cls_list = list(traj_scene_cls_list.values())    

    for i in tqdm(range(len(traj_scene_cls_list)), desc='    Building All Data for %s'%traj_scene_cls_list[0]['sceneId']):
    # for i in range(len(traj_scene_cls_list)):
        num_ped = len(traj_scene_cls_list[i]['traj_raw_xyt'])
        traj_scene_cls_list[i]['num_ped'] = num_ped

        traj_raw_xyt = torch.stack(traj_scene_cls_list[i]['traj_raw_xyt'], dim=0).permute(2,0,1).reshape(1,2,num_ped,traj_len)
        traj_raw_xyt = torch.cat((traj_raw_xyt,torch.linspace(0,traj_len-1,traj_len).reshape(1,1,1,traj_len).repeat(1,1,num_ped,1)), dim=1)
        traj_scene_cls_list[i]['traj_raw_xyt'] = traj_raw_xyt

        traj_raw_vel = traj_raw_xyt[:,:,:,1:] - traj_raw_xyt[:,:,:,:-1]
        traj_raw_vel = torch.cat((traj_raw_vel[:,:,:,0:1],traj_raw_vel),dim=3)
        traj_scene_cls_list[i]['traj_raw_vel'] = traj_raw_vel[:,:2,:,:]

        build_center_scenes(traj_scene_cls_list[i])

        cell_map_xy_goal_gt, cell_map_xy_goal_mask, global_anchor_coord_xyt = build_global_cls_gt_mask_xy(traj_scene_cls_list[i])
        traj_scene_cls_list[i]['anchor_goal_gt'] = cell_map_xy_goal_gt
        traj_scene_cls_list[i]['anchor_goal_gtmask'] = cell_map_xy_goal_mask
        traj_scene_cls_list[i]['anchor_goal_raw_vel'] = global_anchor_coord_xyt

        cell_map_xy_pred_gt, cell_map_xy_pred_mask, local_anchor_coord_xyt = build_local_cls_gt_mask_xy(traj_scene_cls_list[i])
        traj_scene_cls_list[i]['anchor_step_gt'] = cell_map_xy_pred_gt
        traj_scene_cls_list[i]['anchor_step_gtmask'] = cell_map_xy_pred_mask
        traj_scene_cls_list[i]['anchor_step_raw_vel'] = local_anchor_coord_xyt


    return traj_scene_cls_list





with open('raw/train.pkl', 'rb') as f:
    data_raw = pickle5.load(f)

data_raw_aug, scene_img_dict_aug, scene_img_dict = augment_data(data_raw)
resize(scene_img_dict_aug, factor=0.25)
pad(scene_img_dict_aug, division_factor=32)  # make sure that image shape is divisible by 32, for UNet segmentation
preprocess_image_for_segmentation(scene_img_dict_aug)

# traj_scene_cls_list = build_traj_scene_cls_split(data_raw, scene_img_dict)  
traj_scene_cls_list_aug = build_traj_scene_cls_split(data_raw_aug, scene_img_dict_aug)                

with open('%s_train.pkl'%config.dataset_name, 'wb') as f:
    pickle.dump(traj_scene_cls_list_aug, f, protocol=pickle.HIGHEST_PROTOCOL)



with open('raw/test.pkl', 'rb') as f:
    data_raw = pickle5.load(f)

data_raw_aug, scene_img_dict_aug, scene_img_dict = augment_data(data_raw)
resize(scene_img_dict, factor=0.25)
pad(scene_img_dict, division_factor=32)  # make sure that image shape is divisible by 32, for UNet segmentation
preprocess_image_for_segmentation(scene_img_dict)

traj_scene_cls_list = build_traj_scene_cls_split(data_raw, scene_img_dict)  
# traj_scene_cls_list_aug = build_traj_scene_cls_split(data_raw_aug, scene_img_dict_aug)                

with open('%s_test.pkl'%config.dataset_name, 'wb') as f:
    pickle.dump(traj_scene_cls_list, f, protocol=pickle.HIGHEST_PROTOCOL)

