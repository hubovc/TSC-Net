import torch
import numpy as np
import os
import cv2
import pickle

from models.dataset import SceneDataLoader, SceneDataset
from models.utils import *

def build_data_list(test_dataset):
    scene_name_list = []
    start_frame_list = []
    traj_list = []
    for sample in test_dataset.traj_scene_sample_list:
        # for p in range(sample['num_ped']):
        scene_name_list.append(sample['sceneId'])
        start_frame_list.append(sample['start_frame'])

    return scene_name_list, start_frame_list

def find_best_traj_idx(gt_traj, pred_traj):
    gt_traj = gt_traj[:,:,:,8:]
    traj_dist = euclidean_distance(gt_traj, pred_traj) # nb x np x nf
    ade = traj_dist.mean(dim=3).reshape(-1) # nb x np
    best_b_idx = ade.argmin()
    best_ade = ade[best_b_idx]

    return best_b_idx, best_ade

def visualize_goal_one_ped(im, gt_traj, pred_goals, obs_length):
    im = cv2.circle(im, (gt_traj[0,0],gt_traj[0,1]), 5, [0,0,255], -1 )
    for f in range(len(gt_traj)-1):
        color = [0,0,255] if f < obs_length else [255,0,255]
        im = cv2.circle(im, (gt_traj[f+1,0],gt_traj[f+1,1]), 5, color, -1 )
        im = cv2.line(im, (gt_traj[f,0],gt_traj[f,1]), (gt_traj[f+1,0],gt_traj[f+1,1]), color, 2)

    for s in range(len(pred_goals)):
        im = cv2.circle(im, (pred_goals[s,0],pred_goals[s,1]), 5, [255,0,0], -1 )

    # im = cv2.putText(im, '%0.2f'%min_dist, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,0,0], 3, 2)

    return im

def visualize_one_traj(img, traj, color):
    img = cv2.circle(img, (traj[0,0],traj[0,1]), 5, color, -1 )
    for f in range(len(traj)-1):
        img = cv2.circle(img, (traj[f+1,0],traj[f+1,1]), 5, color, -1 )
        img = cv2.line(img, (traj[f,0],traj[f,1]), (traj[f+1,0],traj[f+1,1]), color, 2)

    return img

def visualize_one_goal(img, goal, color):
    img = cv2.circle(img, (goal[0,0],goal[0,1]), 5, color, -1 )
    return img

def visualize_one_text(img, best_ade):
    img = cv2.putText(img, 'max len = %d'%best_ade, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,0,0], 3, 2)
    return img

def visualize_one_ped(img, gt_traj, pred_traj, rough_goal):
    img = visualize_one_traj(img, pred_traj.t(), [255,0,0] )
    img = cv2.line(img, (gt_traj.t()[7,0],gt_traj.t()[7,1]), (pred_traj.t()[0,0],pred_traj.t()[0,1]), [255,0,0], 2)

    img = visualize_one_traj(img, gt_traj[:,:8].t(), [0,0,255] )
    img = visualize_one_traj(img, gt_traj[:,8:].t(), [255,0,255] )
    img = cv2.line(img, (gt_traj.t()[7,0],gt_traj.t()[7,1]), (gt_traj.t()[8,0],gt_traj.t()[8,1]), [255,0,255], 2)

    img = visualize_one_goal(img, rough_goal.t(), [0,255,0])

    return img










def visualize_one_epoch(args, epoch):
    vis_dir = os.path.join(args.visualization_dir,args.dataset_name)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    result_dir =  os.path.join(args.result_dir,args.dataset_name)
    result_file = os.path.join(result_dir,'TSC_Net_{}_Ep_{}.res'.format(args.dataset_name,epoch))

    with open(result_file, 'rb' ) as file:
        all_outputs = pickle.load(file)

    goal_pred_shift, future_pred_shift, traj_gt_shift, all_traj_gt_ref, num_ped_list = all_outputs

    test_dataset = SceneDataset(args, 'test')

    scene_name_list, start_frame_list = build_data_list(test_dataset)

    # print(goal_pred_shift.size(), future_pred_shift.size(), traj_gt_shift.size())
    # print(len(scene_name_list), len(start_frame_list), len(num_ped_list), sum(num_ped_list))
    # tmp = input()

    all_ped_idx = 0
    for scene_idx in range(len(scene_name_list)):
        # print(len(scene_name_list))
        # tmp = input()
        cur_scene = scene_name_list[scene_idx]
        cur_sf = '%07d'%start_frame_list[scene_idx]
        image_path = 'data/keyframes/' + cur_scene + '/' + cur_sf + '.jpg'

        img = cv2.imread(image_path)

        ade_list = []
        # print(scene_idx)
        for ped_idx in range(num_ped_list[scene_idx]):
            gt_traj = traj_gt_shift[:,:,all_ped_idx:all_ped_idx+1]
            pred_traj = future_pred_shift[:,:,all_ped_idx:all_ped_idx+1]
            rough_goal = goal_pred_shift[:,:,all_ped_idx:all_ped_idx+1]

            traj_gt_ref = all_traj_gt_ref[:,:,all_ped_idx:all_ped_idx+1]

            best_b_idx, best_ade = find_best_traj_idx(gt_traj, pred_traj)
            ade_list.append(best_ade)

            one_gt_traj = gt_traj[best_b_idx,:,0] + traj_gt_ref[0,:,0,:].repeat(1,20)            
            one_pred = pred_traj[best_b_idx,:,0] + traj_gt_ref[0,:,0,:].repeat(1,12)
            one_rough_goal = rough_goal[best_b_idx,:,0] + traj_gt_ref[0,:,0,:]
            img = visualize_one_ped(img, one_gt_traj, one_pred, one_rough_goal)

            all_ped_idx = all_ped_idx + 1

        # img = visualize_one_text(img, max(ade_list))
        vis_im_path = os.path.join(vis_dir,'{}_{}.jpg'.format(cur_scene,int(cur_sf)))
        cv2.imwrite(vis_im_path, img) 

        print(vis_im_path)

        # tmp = input()


