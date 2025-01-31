from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import os

import random
import math
import pickle

from models.utils import *

class SceneDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=False,
                 **kwargs):
        super(SceneDataLoader, self).__init__(
            dataset=dataset, 
            batch_size=1, 
            shuffle=shuffle, 
            num_workers=0,
            collate_fn=self._collate_fn, 
            pin_memory=pin_memory, 
            drop_last=drop_last,
            **kwargs
        )

    def _collate_fn(self, batch_raw):
        batch = {}
        for key in batch_raw[0].keys():
            batch[key] = batch_raw[0][key]

        return batch


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, args, phase, test_train=False ):
        assert phase == 'train' or phase == 'test', "Phase must be train or test."
        assert args.dataset_name == 'SDD' or args.dataset_name == 'ETH', "Dataset must be SDD or ETH."
        assert args.map_type == 'semantic' or args.map_type == 'segmentation', "Map type must be semantic or segmentation."

        data_file_name = '{}_{}.pkl'.format(args.dataset_name,phase)
        if test_train:
            data_file_name = '{}_{}.pkl'.format(args.dataset_name,'train')
            
        with open(os.path.join(args.data_dir,data_file_name), 'rb' ) as f:
            self.traj_scene_sample_list = pickle.load(f)

        self.args = args
        self.phase = phase
        self.manual_batch_size = args.manual_batch_size

        self.traj_norm_factor = args.traj_norm
        self.obs = args.obs_length
        self.clip_length = self.args.clip_length
        self.pred_dims = args.models.traj.prediction_dims

        self.shuffle = True if phase == 'train' else False

        if phase == 'train':
            self.build_train_batch_list()
        else:
            self.build_test_batch_list()
        # self.pe = positional_encoding_1d(64, self.args.clip_length)

        self.num_scene_samples = len(self.traj_scene_sample_list)
        self.num_ped_samples = sum([traj_scene['num_ped'] for traj_scene in self.traj_scene_sample_list])

    def __len__(self):
        return self.n_samples

    def build_train_batch_list(self):
        self.traj_scene_batch_list = [[]]

        if self.shuffle:
            random.shuffle(self.traj_scene_sample_list)

        batch_count = 0
        for traj_scene in self.traj_scene_sample_list:
            num_ped = traj_scene['num_ped']
            if batch_count != 0 and batch_count + num_ped > self.manual_batch_size * 2:
                self.traj_scene_batch_list.append([])
                batch_count = 0

            self.traj_scene_batch_list[-1].append(traj_scene)
            batch_count = batch_count + num_ped

            if batch_count > self.manual_batch_size:
                self.traj_scene_batch_list.append([])
                batch_count = 0

        if len(self.traj_scene_batch_list[-1]) == 0:
            self.traj_scene_batch_list.pop()

        self.n_samples = len(self.traj_scene_batch_list)

    def build_test_batch_list(self):
        self.traj_scene_batch_list = []      

        for traj_scene in self.traj_scene_sample_list:
            self.traj_scene_batch_list.append([traj_scene])

        self.n_samples = len(self.traj_scene_batch_list)

    def build_batch(self, traj_scene_batch_raw):
        traj_scene_batch = {}

        # concat to list: frame_idx_list, scene_Id, start_frame, num_ped, scene_map
        # concat to torch tensor: xy_raw, xy_step,
        #                         cell_map_xy_(goal_gt, goal_mask, pred_gt, pred_mask)

        traj_scene_batch['frame_idx_list'] = [traj_scene_sample['frame_idx_list'] for traj_scene_sample in traj_scene_batch_raw]
        traj_scene_batch['start_frame']    = [traj_scene_sample['start_frame'] for traj_scene_sample in traj_scene_batch_raw]
        traj_scene_batch['scene_map']      = [traj_scene_sample['scene_map'] for traj_scene_sample in traj_scene_batch_raw]
        traj_scene_batch['sceneId']        = [traj_scene_sample['sceneId'] for traj_scene_sample in traj_scene_batch_raw]
        traj_scene_batch['num_ped']        = [traj_scene_sample['num_ped'] for traj_scene_sample in traj_scene_batch_raw]
        np_all = sum(traj_scene_batch['num_ped'])

        traj_scene_batch['traj_raw_xyt'] = torch.cat([traj_scene_sample['traj_raw_xyt'] for traj_scene_sample in traj_scene_batch_raw], dim=2)
        traj_scene_batch['traj_raw_vel'] = torch.cat([traj_scene_sample['traj_raw_vel'] for traj_scene_sample in traj_scene_batch_raw], dim=2)

        traj_scene_batch['anchor_goal_gt']     = torch.cat([traj_scene_sample['anchor_goal_gt'] for traj_scene_sample in traj_scene_batch_raw], dim=0)
        traj_scene_batch['anchor_goal_gtmask'] = torch.cat([traj_scene_sample['anchor_goal_gtmask'] for traj_scene_sample in traj_scene_batch_raw], dim=0)
        traj_scene_batch['anchor_step_gt']     = torch.cat([traj_scene_sample['anchor_step_gt'] for traj_scene_sample in traj_scene_batch_raw], dim=0)
        traj_scene_batch['anchor_step_gtmask'] = torch.cat([traj_scene_sample['anchor_step_gtmask'] for traj_scene_sample in traj_scene_batch_raw], dim=0)
        
        traj_scene_batch['anchor_goal_gt']     = traj_scene_batch['anchor_goal_gt'].permute(1,0,2,3).reshape(1,self.pred_dims[-1],np_all,-1)
        traj_scene_batch['anchor_goal_gtmask'] = traj_scene_batch['anchor_goal_gtmask'].permute(1,0,2,3).reshape(1,self.pred_dims[-1],np_all,-1)
        traj_scene_batch['anchor_step_gt']     = traj_scene_batch['anchor_step_gt'].permute(2,0,1,3,4).reshape(1,self.pred_dims[-1],np_all,self.clip_length,-1)
        traj_scene_batch['anchor_step_gtmask'] = traj_scene_batch['anchor_step_gtmask'].permute(2,0,1,3,4).reshape(1,self.pred_dims[-1],np_all,self.clip_length,-1)

        traj_scene_batch['anchor_goal_raw_vel'] = traj_scene_batch_raw[0]['anchor_goal_raw_vel']
        traj_scene_batch['anchor_step_raw_vel'] = traj_scene_batch_raw[0]['anchor_step_raw_vel']

        traj_raw_xyt = traj_scene_batch['traj_raw_xyt']

        nb,nd,np,nf = traj_scene_batch['traj_raw_vel'].size()
        traj_scene_batch['traj_raw_vel'] = torch.cat((traj_scene_batch['traj_raw_vel'],torch.ones(nb,1,np,nf)), dim=1)

        traj_scene_batch['traj_raw_ref'] = traj_raw_xyt[:,:,:,self.obs-1:self.obs]
        traj_scene_batch['traj_raw_shift'] = traj_raw_xyt - traj_scene_batch['traj_raw_ref'].repeat(1,1,1,self.clip_length)        

        scene_map = traj_scene_batch['scene_map']
        scenes_global = [] 
        for scene_idx, np in enumerate(traj_scene_batch['num_ped']):
            for p in range(np):
                scenes_global.append(scene_map[scene_idx])

        scene_goal_raw, scene_step_raw = self.load_scenes(traj_scene_batch_raw)

        traj_scene_batch['scenes_global'] = scenes_global

        traj_scene_batch['traj_raw_xyt'] = traj_scene_batch['traj_raw_xyt'].cuda()
        traj_scene_batch['traj_raw_vel'] = traj_scene_batch['traj_raw_vel'].cuda()
        traj_scene_batch['traj_raw_ref'] = traj_scene_batch['traj_raw_ref'].cuda()
        traj_scene_batch['traj_raw_shift'] = traj_scene_batch['traj_raw_shift'].cuda()

        traj_scene_batch['anchor_goal_raw_vel'] = traj_scene_batch['anchor_goal_raw_vel'].cuda()
        traj_scene_batch['anchor_step_raw_vel'] = traj_scene_batch['anchor_step_raw_vel'].cuda()

        traj_scene_batch['anchor_goal_gt']     = traj_scene_batch['anchor_goal_gt'].cuda()
        traj_scene_batch['anchor_goal_gtmask'] = traj_scene_batch['anchor_goal_gtmask'].cuda()
        traj_scene_batch['anchor_step_gt']     = traj_scene_batch['anchor_step_gt'].cuda()
        traj_scene_batch['anchor_step_gtmask'] = traj_scene_batch['anchor_step_gtmask'].cuda()

        traj_scene_batch['anchor_goal_gtmask'][:,0:1] = traj_scene_batch['anchor_goal_gtmask'][:,0:1] * self.args.models.goal.recon_conf_weight
        traj_scene_batch['anchor_goal_gtmask'][:,1:] = traj_scene_batch['anchor_goal_gtmask'][:,1:] * self.args.models.goal.recon_offset_weight
        traj_scene_batch['anchor_step_gtmask'][:,0:1] = traj_scene_batch['anchor_step_gtmask'][:,0:1] * self.args.models.traj.conf_weight
        traj_scene_batch['anchor_step_gtmask'][:,1:] = traj_scene_batch['anchor_step_gtmask'][:,1:] * self.args.models.traj.offset_weight

        traj_scene_batch['scene_goal_raw'] = scene_goal_raw.cuda()
        traj_scene_batch['scene_step_raw'] = scene_step_raw.cuda()


        return traj_scene_batch

    def load_scenes(self, traj_scene_batch_raw):
        scene_goal_raw_list = []
        scene_step_raw_list = []

        for traj_scene in traj_scene_batch_raw:
            start_frame = traj_scene['start_frame']
            sceneId = traj_scene['sceneId']       

            scene_file_name = '%s/center_scenes/%s_%05d.pkl'%(self.args.data_dir,sceneId,start_frame)
            with open(scene_file_name, 'rb' ) as f:
                one_scene_dict = pickle.load(f)

            scene_goal_raw_list.append(one_scene_dict['scene_goal_raw'])
            scene_step_raw_list.append(one_scene_dict['scene_step_raw'])

        scene_goal_raw = torch.cat(scene_goal_raw_list, dim=0)
        scene_step_raw = torch.cat(scene_step_raw_list, dim=0)

        return scene_goal_raw, scene_step_raw

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        traj_scene_batch_raw = self.traj_scene_batch_list[index]
        traj_scene_batch = self.build_batch(traj_scene_batch_raw)

        return traj_scene_batch
    