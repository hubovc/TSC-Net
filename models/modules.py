import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

import time

from models.utils import *
from models.layers import MulitiHeadAttention, FFN, MLP, Backbone, LayerNorm4D

class FeatureBuilding(nn.Module):
    def __init__(self, args):
        super(FeatureBuilding, self).__init__()

        self.args = args
        self.traj_norm_factor = args.traj_norm
        self.obs = args.obs_length
        self.future = args.future_length
        self.clip_length = args.clip_length
        self.local_anchor_size = args.local_anchor_size

        self.traj_shift_emb = MLP(args.models.feat.traj_shift_emb_dims)
        self.traj_ref_emb = MLP(args.models.feat.traj_ref_emb_dims)
        self.traj_vel_emb = MLP(args.models.feat.traj_vel_emb_dims)
        self.scene_emb = Backbone(args)

        self.history_encoder = HistoryEncoder(args) 
        self.goal_condition_encoder = GoalConditionEncoder(args) 

        self.n_sample_cvae = args.num_sample_cvae
        self.n_sample_cell = args.num_sample_cell

    def build_spatial_n_temporal_mask(self, num_ped_list):
        np_all = sum(num_ped_list)

        spatial_mask = torch.zeros(np_all,np_all).cuda()

        np_acc = 0
        for np in num_ped_list:
            spatial_mask[np_acc:np_acc+np,np_acc:np_acc+np] = 1.
            np_acc = np_acc + np

        nf_all = self.clip_length
        temporal_mask = torch.ones(nf_all,nf_all).tril().cuda()
        temporal_mask[:self.obs,:self.obs] = 1
        # print(temporal_mask)
        return spatial_mask, temporal_mask

    def build_traj_net_training_mask(self):
        f_obs = self.obs
        f_future = self.future
        f_all_obs = self.clip_length - 1
        n_cell = self.local_anchor_size * self.local_anchor_size

        self_attn_mask = torch.eye(f_future).reshape(f_future,1,f_future,1).repeat(1,n_cell,1,n_cell).reshape(f_future*n_cell,f_future*n_cell).cuda()
        cross_attn_mask = torch.ones(f_future,f_all_obs).tril(diagonal=f_obs-1).cuda()
        cross_attn_mask = cross_attn_mask.reshape(f_future,1,f_all_obs).repeat(1,n_cell,1).reshape(f_future*n_cell,f_all_obs)

        return self_attn_mask, cross_attn_mask

    def build_goal_scene(self, traj_raw_xyt, scenes_global):
        scene_goal_list = []
        goal_center_pos = traj_raw_xyt[:,:2,:,:]
        for p in range(traj_raw_xyt.size(2)):
            scene_goal_list_p = []
            for b in range(traj_raw_xyt.size(0)):
                one_scene_goal_raw = crop_and_pad(goal_center_pos[b,:,p,0], scenes_global[p], self.args.global_size, self.args.resize_factor)
                scene_goal_list_p.append(one_scene_goal_raw)
            scene_goal_list_p = torch.stack(scene_goal_list_p, dim=0).cuda()
            scene_goal_list.append(scene_goal_list_p)
        scene_goal_raw = torch.stack(scene_goal_list, dim=1).cuda()
        # nb x np x c x h x w
        return scene_goal_raw

    def build_step_scene(self, traj_raw_xyt, scenes_global): 
        scene_step_list = []
        for b in range(traj_raw_xyt.size(0)):
            scene_step_b_list = []
            for p in range(traj_raw_xyt.size(2)):
                scene_step_p_list = []
                for f in range(traj_raw_xyt.size(3)):
                    step_center_pos = traj_raw_xyt[0,:2,p,f]
                    one_scene_step_raw = crop_and_pad(step_center_pos, scenes_global[p], self.args.local_size * 3, self.args.resize_factor)
                    scene_step_p_list.append(one_scene_step_raw)

                scene_step_b_list.append(torch.stack(scene_step_p_list,0)) # T x 15 x 15 x 7
            scene_step_list.append(torch.stack(scene_step_b_list,0)) # T x 15 x 15 x 7
        scene_step_raw = torch.stack(scene_step_list,0).cuda()

        return scene_step_raw
    
    def build_all_feat_gt(self, batch_data, spatial_mask, temporal_mask):
        # all attn feat
        traj_raw_xyt = batch_data['traj_raw_xyt']
        traj_raw_vel = batch_data['traj_raw_vel']
        traj_raw_ref = batch_data['traj_raw_ref']
        traj_raw_shift = batch_data['traj_raw_shift']
        scene_step_raw = batch_data['scene_step_raw']

        nb, nd, np, nf = traj_raw_xyt.size()

        traj_feat_ref = self.traj_ref_emb(traj_raw_ref)
        traj_feat_shift = self.traj_shift_emb(traj_raw_shift)
        traj_feat_vel = self.traj_vel_emb(traj_raw_vel[:,:2])

        scene_step_feat_3x3 = self.scene_emb.scene_step_emb_to_3x3(scene_step_raw) # n x f x d x 3 x 3
        scene_step_feat = self.scene_emb.scene_step_emb_to_1x1(scene_step_feat_3x3) # 1 x d x n x f

        cell_feat = torch.cat([traj_feat_ref.repeat(1,1,1,nf),traj_feat_shift,traj_feat_vel,scene_step_feat],dim=1)

        cell_attn = self.history_encoder(cell_feat, cell_feat, spatial_mask=spatial_mask, temporal_mask=temporal_mask)

        np, nf, nd, nh, nw = scene_step_feat_3x3.size()
        anchor_step_feat_scene = scene_step_feat_3x3.reshape(1,np,nf,nd,nh*nw).permute(0,3,1,2,4) # nb x nd x np x nf x (nh*nw)
        

        # build anchor step feat from ground truth
        nb, nd, np, nf = traj_raw_shift[:,:,:,self.obs:].size()
        anchor_step_raw_vel = batch_data['anchor_step_raw_vel'].reshape(1,3,1,1,-1).repeat(1,1,np,nf,1) # 1 x 3 x np x nf(12) x 9
        nhnw = anchor_step_raw_vel.size(-1)
        traj_raw_shift_prev = traj_raw_shift[:,:,:,self.obs-1:-1].reshape(nb,nd,np,nf,1).repeat(1,1,1,1,nhnw)
        anchor_step_raw_shift = traj_raw_shift_prev + anchor_step_raw_vel

        anchor_step_feat_ref = traj_feat_ref.reshape(nb,-1,np,1,1).repeat(1,1,1,nf,nhnw)
        anchor_step_feat_vel = self.traj_vel_emb(anchor_step_raw_vel[:,:2].reshape(nb,2,np*nf,nhnw)).reshape(nb,-1,np,nf,nhnw)
        anchor_step_feat_shift = self.traj_shift_emb(anchor_step_raw_shift.reshape(nb,nd,np*nf,nhnw)).reshape(nb,-1,np,nf,nhnw)

        anchor_step_feat = torch.cat([anchor_step_feat_ref,anchor_step_feat_shift,anchor_step_feat_vel,anchor_step_feat_scene[:,:,:,self.obs:,:]],dim=1)


        # build anchor goal feat from ground truth
        scene_goal_raw = batch_data['scene_goal_raw']
        anchor_goal_raw_vel = batch_data['anchor_goal_raw_vel'][:,:2].reshape(1,2,1,-1).repeat(1,1,np,1) # 1 x 3 x np x 225
        nhnw = anchor_goal_raw_vel.size(-1)
        anchor_goal_raw_shift = torch.cat((anchor_goal_raw_vel,torch.ones(1,1,np,nhnw).cuda()*self.future), dim=1)

        anchor_goal_feat_ref = traj_feat_ref.reshape(nb,-1,np,1).repeat(1,1,1,nhnw)
        anchor_goal_feat_vel = self.traj_vel_emb(anchor_goal_raw_vel)
        anchor_goal_feat_shift = self.traj_shift_emb(anchor_goal_raw_shift)
        anchor_goal_feat_scene = self.scene_emb.scene_goal_emb(scene_goal_raw).permute(1,0,2,3).reshape(1,-1,np,nhnw)

        anchor_goal_feat = torch.cat([anchor_goal_feat_ref,anchor_goal_feat_shift,anchor_goal_feat_vel,anchor_goal_feat_scene],dim=1)


        # build cell goal feat from ground truth
        cell_goal_feat_ref = traj_feat_ref
        cell_goal_feat_shift = traj_feat_shift[:,:,:,-1:]        
        cell_goal_feat_scene = scene_step_feat[:,:,:,-1:]
        cell_goal_feat_vel = self.traj_vel_emb(traj_raw_shift[:,:2,:,-1:])

        cell_goal_feat = torch.cat([cell_goal_feat_ref,cell_goal_feat_shift,cell_goal_feat_vel,cell_goal_feat_scene], dim=1)


        return cell_attn, anchor_step_feat, anchor_goal_feat, cell_goal_feat

    def cnovert_anchor_step_pred_to_traj_training(self, anchor_step_pred, batch_data):
        # In training, nf=12, nb=1
        nb, nd, np, nf, ncell = anchor_step_pred.size()

        anchor_step_raw_vel = batch_data['anchor_step_raw_vel'].reshape(1,3,1,1,-1).repeat(1,1,np,1,1) # 1 x 3 x 1 x 1 x 9
        traj_raw_shift_prev = batch_data['traj_raw_shift'][:,:,:,self.obs-1:-1]

        traj_pred_vel = []
        for f in range(nf):
            max_idx = anchor_step_pred[:,0:1,:,f,:].argmax(dim=3).reshape(np)
            traj_pred_vel_offset = anchor_step_pred[:,1:,torch.LongTensor(range(np)),f,max_idx]
            traj_pred_vel_anchor = anchor_step_raw_vel[:,:2,torch.LongTensor(range(np)),0,max_idx]
            traj_pred_vel_cur = traj_pred_vel_offset + traj_pred_vel_anchor # 1 x 2 x np

            traj_pred_vel_cur = torch.cat((traj_pred_vel_cur,torch.ones(1,1,np).cuda()), dim=1)
            traj_pred_vel.append(traj_pred_vel_cur)

        traj_pred_vel = torch.stack(traj_pred_vel, dim=3) # nb(1) x nd(3) x np x nf(12)
        
        traj_pred_shift = traj_raw_shift_prev + traj_pred_vel
        traj_gt_shift = batch_data['traj_raw_shift'][:,:,:,self.obs:]

        return traj_pred_shift, traj_gt_shift

    def cnovert_anchor_goal_pred_to_traj_training(self, anchor_goal_recon, batch_data):
        nb, nd, np, ncell = anchor_goal_recon.size()
        anchor_goal_raw_vel = batch_data['anchor_goal_raw_vel'].reshape(1,3,1,-1).repeat(1,1,np,1) # 1 x 3 x 1 x 15
        
        max_idx = anchor_goal_recon[:,0:1,:,:].argmax(dim=3).reshape(np)
        goal_pred_shift_offset = anchor_goal_recon[:,1:,torch.LongTensor(range(np)),max_idx]
        goal_pred_shift_anchor = anchor_goal_raw_vel[:,:2,torch.LongTensor(range(np)),max_idx]
        goal_pred_shift = goal_pred_shift_offset + goal_pred_shift_anchor
        goal_pred_shift = goal_pred_shift.unsqueeze(-1)

        goal_gt_shift = traj_gt_shift = batch_data['traj_raw_shift'][:,:2,:,-1:]

        return goal_pred_shift, goal_gt_shift

    def build_all_feat_pred(self, batch_data, spatial_mask, flag_gt_goal, n_sample):
        # obs attn feat
        traj_raw_xyt = batch_data['traj_raw_xyt'][:,:,:,:self.obs]
        traj_raw_vel = batch_data['traj_raw_vel'][:,:,:,:self.obs]
        traj_raw_ref = batch_data['traj_raw_ref']
        traj_raw_shift = batch_data['traj_raw_shift'][:,:,:,:self.obs]
        scene_step_raw = batch_data['scene_step_raw'][:,:self.obs,:,:,:] # np x nf x 7 x nw x nh

        nb, nd, np, nf = traj_raw_xyt.size()

        traj_feat_ref = self.traj_ref_emb(traj_raw_ref)
        traj_feat_shift = self.traj_shift_emb(traj_raw_shift)
        traj_feat_vel = self.traj_vel_emb(traj_raw_vel[:,:2])

        scene_step_feat_3x3 = self.scene_emb.scene_step_emb_to_3x3(scene_step_raw) # n x f x d x 3 x 3
        scene_step_feat = self.scene_emb.scene_step_emb_to_1x1(scene_step_feat_3x3) # 1 x d x n x f

        cell_feat_obs = torch.cat([traj_feat_ref.repeat(1,1,1,nf),traj_feat_shift,traj_feat_vel,scene_step_feat],dim=1)

        cell_attn_obs = self.history_encoder(cell_feat_obs, cell_feat_obs, spatial_mask=spatial_mask)


        # build anchor step feat from ground truth for only the last frame of observation
        anchor_step_raw_vel = batch_data['anchor_step_raw_vel'].reshape(1,3,1,-1).repeat(1,1,np,1) # 1 x 3 x np x 9
        nhnw = anchor_step_raw_vel.size(-1)
        traj_raw_shift_prev = traj_raw_shift[:,:,:,-1:]

        anchor_step_raw_shift = traj_raw_shift_prev + anchor_step_raw_vel

        anchor_step_feat_ref = traj_feat_ref.reshape(nb,-1,np,1).repeat(1,1,1,nhnw)
        anchor_step_feat_vel = self.traj_vel_emb(anchor_step_raw_vel[:,:2])
        anchor_step_feat_shift = self.traj_shift_emb(anchor_step_raw_shift)

        anchor_step_feat_scene = scene_step_feat_3x3[:,-1:].permute(1,2,0,3,4).reshape(1,scene_step_feat_3x3.size(2),np,-1)

        anchor_step_feat = torch.cat([anchor_step_feat_ref,anchor_step_feat_shift,anchor_step_feat_vel,anchor_step_feat_scene],dim=1)
        anchor_step_feat = anchor_step_feat.repeat(n_sample,1,1,1)

        # build anchor goal feat from ground truth
        scene_goal_raw = batch_data['scene_goal_raw']
        anchor_goal_raw_vel = batch_data['anchor_goal_raw_vel'][:,:2].reshape(1,2,1,-1).repeat(1,1,np,1) # 1 x 3 x np x 225
        nhnw = anchor_goal_raw_vel.size(-1)
        anchor_goal_raw_shift = torch.cat((anchor_goal_raw_vel,torch.ones(1,1,np,nhnw).cuda()*self.future), dim=1)

        anchor_goal_feat_ref = traj_feat_ref.reshape(nb,-1,np,1).repeat(1,1,1,nhnw)
        anchor_goal_feat_vel = self.traj_vel_emb(anchor_goal_raw_vel)
        anchor_goal_feat_shift = self.traj_shift_emb(anchor_goal_raw_shift)
        anchor_goal_feat_scene = self.scene_emb.scene_goal_emb(scene_goal_raw).permute(1,0,2,3).reshape(1,-1,np,nhnw)

        anchor_goal_feat = torch.cat([anchor_goal_feat_ref,anchor_goal_feat_shift,anchor_goal_feat_vel,anchor_goal_feat_scene],dim=1)

        if flag_gt_goal:
            cell_goal_feat = build_cell_goal_from_gt(self, traj_feat_ref, traj_feat_shift, scene_step_feat, traj_raw_shift)
        else:
            cell_goal_feat = None

        return cell_feat_obs, cell_attn_obs, anchor_step_feat, anchor_goal_feat, cell_goal_feat, traj_feat_ref

    def build_cell_goal_from_gt(self, traj_feat_ref, traj_feat_shift, scene_step_feat, traj_raw_shift):

        # build cell goal feat from ground truth
        cell_goal_feat_ref = traj_feat_ref
        cell_goal_feat_shift = traj_feat_shift[:,:,:,-1:]        
        cell_goal_feat_scene = scene_step_feat[:,:,:,-1:]
        cell_goal_feat_vel = self.traj_vel_emb(traj_raw_shift[:,:2,:,-1:])

        cell_goal_feat = torch.cat([cell_goal_feat_ref,cell_goal_feat_shift,cell_goal_feat_vel,cell_goal_feat_scene], dim=1)

        return cell_goal_feat

    def build_anchor_goal_attn(self, anchor_goal_feat, cell_feat):
        anchor_goal_attn = self.goal_condition_encoder(anchor_goal_feat, cell_feat )

        return anchor_goal_attn

    def build_dict_update(self, batch_data, cell_feat_history, cell_attn_history, n_sample):
        dict_update = {}
        dict_update['traj_raw_shift_history'] = batch_data['traj_raw_shift'][:,:,:,:self.obs].repeat(n_sample,1,1,1)
        dict_update['cell_feat_history'] = cell_feat_history.repeat(n_sample,1,1,1)
        dict_update['cell_attn_history'] = cell_attn_history.repeat(n_sample,1,1,1)

        return dict_update

    def build_cell_goal_from_pred(self, batch_data, anchor_goal_pred, traj_feat_ref):
        nb, nd, np, ncell = anchor_goal_pred.size()
        anchor_goal_raw_vel = batch_data['anchor_goal_raw_vel'][:,:2].reshape(1,2,1,-1).repeat(1,1,np,1) # 1 x 2 x np x 225

        anchor_goal_pred[:,1:] = anchor_goal_pred[:,1:] + anchor_goal_raw_vel.repeat(nb,1,1,1) # nb x 2 x np x 225
        # anchor_goal_pred[:,1:] = anchor_goal_raw_vel.repeat(nb,1,1,1)

        traj_raw_shift_goal_gt = batch_data['traj_raw_shift'][:,:2,:,-1:]

        # sampling
        traj_raw_vel_goal_pred = []
        sort_idx = (-anchor_goal_pred[:,0,:,:]).sort(dim=-1)[1] # nb x np x 225

        for b in range(nb):
            max_n_goal_shift = [ anchor_goal_pred[b,1:,p,sort_idx[b,p,:self.n_sample_cell]] for p in range(np) ]
            # print(max_n_goal_shift[0].size())
            # tmp = input()
            max_n_goal_shift = torch.stack(max_n_goal_shift, dim=1) # 2 x np x n_max

            traj_raw_vel_goal_pred.append(max_n_goal_shift)

        traj_raw_vel_goal_pred = torch.stack(traj_raw_vel_goal_pred, dim=0) # nb x 2 x np x n_max
        
        nb = nb * self.n_sample_cell
        traj_raw_vel_goal_pred = traj_raw_vel_goal_pred.permute(0,3,1,2).reshape(nb,2,np,1) # nb x 2 x np x 1 

        # compare
        dist = euclidean_distance(traj_raw_vel_goal_pred, traj_raw_shift_goal_gt.repeat(nb,1,1,1)).reshape(nb,np) # nb x np
        dist_sort_idx = dist.sort(dim=0)[1] # nb x np

        traj_raw_vel_goal_pred = [ traj_raw_vel_goal_pred[dist_sort_idx[:,p],:,p,:] for p in range(np) ]
        traj_raw_vel_goal_pred = torch.stack(traj_raw_vel_goal_pred, dim=2) # nb(20) x nd(2) x np x nf(1)

        # build feature
        traj_raw_shift_goal_pred = torch.cat((traj_raw_vel_goal_pred,torch.ones(nb,1,np,1).cuda()*self.future), dim=1)
        traj_raw_xyt_goal_pred = traj_raw_shift_goal_pred + batch_data['traj_raw_ref'].repeat(nb,1,1,1)

        cell_goal_feat_ref = traj_feat_ref.repeat(nb,1,1,1)
        cell_goal_feat_shift = self.traj_shift_emb(traj_raw_shift_goal_pred)
        cell_goal_feat_vel = self.traj_vel_emb(traj_raw_vel_goal_pred)

        scene_step_goal_raw = self.build_step_scene(traj_raw_xyt_goal_pred, batch_data['scenes_global'])
        scene_step_goal_raw = scene_step_goal_raw.squeeze(2)

        cell_goal_feat_scene_3x3 = self.scene_emb.scene_step_emb_to_3x3(scene_step_goal_raw) # nb x np x d x 3 x 3
        cell_goal_feat_scene = self.scene_emb.scene_step_emb_to_1x1(cell_goal_feat_scene_3x3) # 1 x d x nb x np
        cell_goal_feat_scene = cell_goal_feat_scene.permute(2,1,3,0)
        
        cell_goal_feat = torch.cat([cell_goal_feat_ref,cell_goal_feat_shift,cell_goal_feat_vel,cell_goal_feat_scene], dim=1)

        return traj_raw_shift_goal_pred, cell_goal_feat

    def build_cell_goal_from_pred_one_ped(self, batch_data, anchor_goal_pred, traj_feat_ref, p_idx, n_sample_cell):
        nb, nd, np, ncell = anchor_goal_pred.size()
        anchor_goal_raw_vel = batch_data['anchor_goal_raw_vel'][:,:2].reshape(1,2,1,-1) # 1 x 2 x np x 225
        scene_goal_raw = batch_data['scene_goal_raw']

        # scene_goal_conf_mask = scene_goal_raw[p_idx:p_idx+1,1,:,:].reshape(np,15,20,15,20).permute(0,1,3,2,4).reshape(np,ncell,400)
        # # scene_goal_conf_mask = scene_goal_conf_mask.min(dim=-1)[0]
        # scene_goal_conf_mask = scene_goal_conf_mask.mean(dim=-1)
        # scene_goal_conf_mask = scene_goal_conf_mask.reshape(1,1,np,ncell).repeat(nb,1,1,1)
        # # print(anchor_goal_pred.size(), scene_goal_conf_mask.size(), scene_goal_raw.size())
        # # tmp = input()

        # anchor_goal_pred[:,:1] = anchor_goal_pred[:,:1] * scene_goal_conf_mask
        anchor_goal_pred[:,1:] = anchor_goal_pred[:,1:] + anchor_goal_raw_vel.repeat(nb,1,1,1) # nb x 2 x np x 225
        # anchor_goal_pred[:,1:] = anchor_goal_raw_vel.repeat(nb,1,1,1)

        traj_raw_shift_goal_gt = batch_data['traj_raw_shift'][:,:2,p_idx:p_idx+1,-1:]

        # sampling
        traj_raw_vel_goal_pred = []
        sort_idx = (-anchor_goal_pred[:,0,:,:]).sort(dim=-1)[1] # nb x np x 225

        for b in range(nb):
            max_n_goal_shift = [ anchor_goal_pred[b,1:,p,sort_idx[b,p,:n_sample_cell]] for p in range(np) ]
            # print(max_n_goal_shift[0].size())
            # tmp = input()
            max_n_goal_shift = torch.stack(max_n_goal_shift, dim=1) # 2 x np x n_max

            traj_raw_vel_goal_pred.append(max_n_goal_shift)

        traj_raw_vel_goal_pred = torch.stack(traj_raw_vel_goal_pred, dim=0) # nb x 2 x np x n_max
        
        nb = nb * n_sample_cell
        traj_raw_vel_goal_pred = traj_raw_vel_goal_pred.permute(0,3,1,2).reshape(nb,2,np,1) # nb x 2 x np x 1 

        # compare
        dist = euclidean_distance(traj_raw_vel_goal_pred, traj_raw_shift_goal_gt.repeat(nb,1,1,1)).reshape(nb,np) # nb x np
        dist_sort_idx = dist.sort(dim=0)[1] # nb x np

        traj_raw_vel_goal_pred = [ traj_raw_vel_goal_pred[dist_sort_idx[:,p],:,p,:] for p in range(np) ]
        traj_raw_vel_goal_pred = torch.stack(traj_raw_vel_goal_pred, dim=2) # nb(20) x nd(2) x np x nf(1)

        # build feature
        traj_raw_shift_goal_pred = torch.cat((traj_raw_vel_goal_pred,torch.ones(nb,1,np,1).cuda()*self.future), dim=1)
        traj_raw_xyt_goal_pred = traj_raw_shift_goal_pred + batch_data['traj_raw_ref'][:,:,p_idx:p_idx+1,:].repeat(nb,1,1,1)

        cell_goal_feat_ref = traj_feat_ref[:,:,p_idx:p_idx+1].repeat(nb,1,1,1)
        cell_goal_feat_shift = self.traj_shift_emb(traj_raw_shift_goal_pred)
        cell_goal_feat_vel = self.traj_vel_emb(traj_raw_vel_goal_pred)

        # print(batch_data['scenes_global'][0].size())
        # tmp = input()
        scene_step_goal_raw = self.build_step_scene(traj_raw_xyt_goal_pred, [batch_data['scenes_global'][p_idx]])
        scene_step_goal_raw = scene_step_goal_raw.squeeze(2)

        cell_goal_feat_scene_3x3 = self.scene_emb.scene_step_emb_to_3x3(scene_step_goal_raw) # nb x np x d x 3 x 3
        cell_goal_feat_scene = self.scene_emb.scene_step_emb_to_1x1(cell_goal_feat_scene_3x3) # 1 x d x nb x np
        cell_goal_feat_scene = cell_goal_feat_scene.permute(2,1,3,0)
        
        cell_goal_feat = torch.cat([cell_goal_feat_ref,cell_goal_feat_shift,cell_goal_feat_vel,cell_goal_feat_scene], dim=1)

        return traj_raw_shift_goal_pred, cell_goal_feat

    def update_feat_n_dict(self, dict_update, anchor_step_pred_cur, batch_data, traj_feat_ref, spatial_mask ):
        # output updated dict_update, and next frame input query

        traj_raw_shift_history = dict_update['traj_raw_shift_history']
        cell_feat_history = dict_update['cell_feat_history']

        nb, nd, np, ncell = anchor_step_pred_cur.size()
        traj_raw_ref = batch_data['traj_raw_ref']
        anchor_step_raw_vel = batch_data['anchor_step_raw_vel'].reshape(1,3,1,-1).repeat(1,1,np,1)

        traj_raw_vel_cur = []
        for b in range(nb):
            max_idx = anchor_step_pred_cur[b:b+1,0:1,:,:].argmax(dim=3).reshape(-1) # 1 x 1 x np x 1
            one_traj_raw_vel_cur = anchor_step_pred_cur[b:b+1,1:,torch.LongTensor(range(np)),max_idx] + anchor_step_raw_vel[:,:2,torch.LongTensor(range(np)),max_idx]
            # print(one_traj_raw_vel_cur.unsqueeze(-1).size(), torch.ones(1,1,np,1).cuda().size())
            one_traj_raw_vel_cur = torch.cat((one_traj_raw_vel_cur.unsqueeze(-1),torch.ones(1,1,np,1).cuda()), dim=1)
            traj_raw_vel_cur.append(one_traj_raw_vel_cur)

        traj_raw_vel_cur = torch.cat(traj_raw_vel_cur, dim=0)

        traj_raw_shift_cur = traj_raw_vel_cur + traj_raw_shift_history[:,:,:,-1:]
        traj_raw_xyt_cur = traj_raw_shift_cur + traj_raw_ref        

        # compute cell_feat and cell_attn
        scenes_global = batch_data['scenes_global']  
        scene_step_raw_cur = self.build_step_scene(traj_raw_xyt_cur, scenes_global) # nb x np x nf x 7 x nh x nw
        scene_step_raw_cur = scene_step_raw_cur.squeeze(2)

        traj_feat_shift_cur = self.traj_shift_emb(traj_raw_shift_cur)
        traj_feat_vel_cur = self.traj_vel_emb(traj_raw_vel_cur[:,:2])
        traj_feat_ref_cur = traj_feat_ref.repeat(nb,1,1,1)

        scene_step_feat_3x3 = self.scene_emb.scene_step_emb_to_3x3(scene_step_raw_cur) # n x t x d x 3 x 3
        scene_step_feat = self.scene_emb.scene_step_emb_to_1x1(scene_step_feat_3x3) # 1 x nd x nb x np
        scene_step_feat = scene_step_feat.permute(2,1,3,0)

        cell_feat_cur = torch.cat([traj_feat_ref_cur,traj_feat_shift_cur,traj_feat_vel_cur,scene_step_feat],dim=1)

        np, nf_q  = cell_feat_cur.size()[2:4]
        np, nf_kv = cell_feat_history.size()[2:4]
        spatial_temporal_mask = spatial_mask.reshape(np,1,np,1).repeat(1,nf_q,1,nf_kv).reshape(np*nf_q,np*nf_kv)

        cell_attn_cur = self.history_encoder.inference(cell_feat_cur, cell_feat_history, spatial_mask=spatial_mask)

        # compute anchor_step_feat
        anchor_step_raw_vel = batch_data['anchor_step_raw_vel'].reshape(1,3,1,-1).repeat(nb,1,np,1)
        nhnw = anchor_step_raw_vel.size(3)

        anchor_step_raw_shift = traj_raw_shift_cur.repeat(1,1,1,nhnw) + anchor_step_raw_vel

        anchor_step_feat_ref = traj_feat_ref.repeat(nb,1,1,nhnw)
        anchor_step_feat_vel = self.traj_vel_emb(anchor_step_raw_vel[:,:2])
        anchor_step_feat_shift = self.traj_shift_emb(anchor_step_raw_shift)
        anchor_step_feat_scene = scene_step_feat_3x3.reshape(nb,-1,np,nhnw)

        anchor_step_feat = torch.cat([anchor_step_feat_ref,anchor_step_feat_shift,anchor_step_feat_vel,anchor_step_feat_scene],dim=1)  

        # update final output prediction
        dict_update['traj_raw_shift_history'] = torch.cat((dict_update['traj_raw_shift_history'],traj_raw_shift_cur), dim=3)
        dict_update['cell_feat_history'] = torch.cat((dict_update['cell_feat_history'],cell_feat_cur), dim=3)
        dict_update['cell_attn_history'] = torch.cat((dict_update['cell_attn_history'],cell_attn_cur), dim=3)

        return anchor_step_feat


















class HistoryEncoder(nn.Module):
    def __init__(self, args):
        super(HistoryEncoder, self).__init__()

        self.n_layers = args.models.feat.n_layers
        n_heads = args.models.feat.n_heads
        dropout = args.models.feat.dropout

        enc_dims = args.models.feat.encoder_att_dims
        ffn_dims = args.models.feat.encoder_ffn_dims

        self.temporal_self_attn_layers = nn.ModuleList()
        self.temporal_self_attn_norm_layers = nn.ModuleList() 
        self.temporal_self_attn_ffn_layers = nn.ModuleList()
        for l in range(self.n_layers):
            self.temporal_self_attn_layers.append(MulitiHeadAttention(enc_dims[0], enc_dims[1], n_heads, dropout=dropout))
            self.temporal_self_attn_norm_layers.append(LayerNorm4D(enc_dims[0]))
            self.temporal_self_attn_ffn_layers.append(FFN(ffn_dims[0], ffn_dims[1], dropout=dropout))

        self.ped_self_attn_layers = nn.ModuleList()
        self.ped_self_attn_norm_layers = nn.ModuleList() 
        self.ped_self_attn_ffn_layers = nn.ModuleList()
        for l in range(self.n_layers):
            self.ped_self_attn_layers.append(MulitiHeadAttention(enc_dims[0], enc_dims[1], n_heads, dropout=dropout))
            self.ped_self_attn_norm_layers.append(LayerNorm4D(enc_dims[0]))
            self.ped_self_attn_ffn_layers.append(FFN(ffn_dims[0], ffn_dims[1], dropout=dropout))

    def forward(self, xq, xkv, spatial_mask=None, temporal_mask=None):
        nb, nd, np, nfq = xq.size()
        nb, nd, np, nfkv = xkv.size()

        # temporal self-attention
        xq = xq.permute(0,2,1,3).reshape(nb*np,nd,1,nfq)
        xkv_t = xkv.permute(0,2,1,3).reshape(nb*np,nd,1,nfkv)
        # print(xq.size(), xkv_t.size(), spatial_mask.size())
        for l in range(self.n_layers):
            att_x = self.temporal_self_attn_layers[l](xq,xkv_t,xkv_t,mask=temporal_mask)
            xq = att_x + xq
            xq = self.temporal_self_attn_norm_layers[l](xq)
            xq = self.temporal_self_attn_ffn_layers[l](xq)
        xq = xq.reshape(nb,np,nd,nfq).permute(0,2,1,3)

        # pedestrian self-attention, i.e. interactions
        xq = xq.permute(0,3,1,2).reshape(nb*nfq,nd,1,np)
        xkv_p = xkv.permute(0,3,1,2).reshape(nb*nfkv,nd,1,np)

        # print(xq.size(), xkv_p.size(), spatial_mask.size())
        for l in range(self.n_layers):
            att_x = self.ped_self_attn_layers[l](xq,xkv_p,xkv_p,mask=spatial_mask)
            xq = att_x + xq
            xq = self.ped_self_attn_norm_layers[l](xq)
            xq = self.ped_self_attn_ffn_layers[l](xq)
        xq = xq.reshape(nb,nfq,nd,np).permute(0,2,3,1)

        return xq

    def inference(self, xq, xkv, spatial_mask=None):
        nb, nd, np, nfq = xq.size()
        nb, nd, np, nfkv = xkv.size()

        # temporal self-attention
        xq = xq.permute(0,2,1,3).reshape(nb*np,nd,1,nfq)
        xkv_t = xkv.permute(0,2,1,3).reshape(nb*np,nd,1,nfkv)
        # print(xq.size(), xkv_t.size(), spatial_mask.size())
        for l in range(self.n_layers):
            att_x = self.temporal_self_attn_layers[l](xq,xkv_t,xkv_t)
            xq = att_x + xq
            xq = self.temporal_self_attn_norm_layers[l](xq)
            xq = self.temporal_self_attn_ffn_layers[l](xq)
        xq = xq.reshape(nb,np,nd,nfq).permute(0,2,1,3)

        # pedestrian self-attention, i.e. interactions
        xq = xq.permute(0,3,1,2).reshape(nb*nfq,nd,1,np)
        # xkv_p = xkv.permute(0,3,1,2).reshape(nb*nfkv,nd,1,np)

        # print(xq.size(), xkv_p.size(), spatial_mask.size())
        for l in range(self.n_layers):
            att_x = self.ped_self_attn_layers[l](xq,xq,xq,mask=spatial_mask)
            xq = att_x + xq
            xq = self.ped_self_attn_norm_layers[l](xq)
            xq = self.ped_self_attn_ffn_layers[l](xq)
        xq = xq.reshape(nb,nfq,nd,np).permute(0,2,3,1)

        return xq


class GoalConditionEncoder(nn.Module):
    def __init__(self, args):
        super(GoalConditionEncoder, self).__init__()

        self.n_layers = args.models.feat.n_layers
        n_heads = args.models.feat.n_heads
        dropout = args.models.feat.dropout

        enc_dims = args.models.feat.encoder_att_dims
        ffn_dims = args.models.feat.encoder_ffn_dims

        self.goal_cell_self_att_layers = nn.ModuleList()
        self.goal_cell_self_att_norm_layers = nn.ModuleList() 
        self.goal_cell_self_att_ffn_layers = nn.ModuleList()
        for l in range(self.n_layers):
            self.goal_cell_self_att_layers.append(MulitiHeadAttention(enc_dims[0], enc_dims[1], n_heads, dropout=dropout))
            self.goal_cell_self_att_norm_layers.append(LayerNorm4D(enc_dims[0]))
            self.goal_cell_self_att_ffn_layers.append(FFN(ffn_dims[0], ffn_dims[1], dropout=dropout))

        self.goal_cell_cross_att_layers = nn.ModuleList()
        self.goal_cell_cross_att_norm_layers = nn.ModuleList() 
        self.goal_cell_cross_att_ffn_layers = nn.ModuleList()
        for l in range(self.n_layers):
            self.goal_cell_cross_att_layers.append(MulitiHeadAttention(enc_dims[0], enc_dims[1], n_heads, dropout=dropout))
            self.goal_cell_cross_att_norm_layers.append(LayerNorm4D(enc_dims[0]))
            self.goal_cell_cross_att_ffn_layers.append(FFN(ffn_dims[0], ffn_dims[1], dropout=dropout))

    def forward(self, xq, xkv, mask=None):
        # goal cell self-attention without interactions
        nb,nd,np,n_cell = xq.size()        
        for l in range(self.n_layers):
            xq = xq.permute(0,2,1,3).reshape(nb*np,nd,1,n_cell)
            att_x = self.goal_cell_self_att_layers[l](xq,xq,xq)
            xq = att_x + xq
            xq = self.goal_cell_self_att_norm_layers[l](xq)
            xq = self.goal_cell_self_att_ffn_layers[l](xq)
            xq = xq.reshape(nb,np,nd,n_cell).permute(0,2,1,3)

        # goal cell cross attention (attending observation) without interactions
        nb,nd,np,nf = xkv.size()        
        xkv = xkv.permute(0,2,1,3).reshape(nb*np,nd,1,nf)
        for l in range(self.n_layers):
            xq = xq.permute(0,2,1,3).reshape(nb*np,nd,1,n_cell)
            att_x = self.goal_cell_cross_att_layers[l](xq,xkv,xkv)
            xq = att_x + xq
            xq = self.goal_cell_cross_att_norm_layers[l](xq)
            xq = self.goal_cell_cross_att_ffn_layers[l](xq)
            xq = xq.reshape(nb,np,nd,n_cell).permute(0,2,1,3)

        # xq shape: b(1) x d(256) x n x n_cell(15x15)
        return xq

class StepDecoder(nn.Module):
    def __init__(self, args):
        super(StepDecoder, self).__init__()

        self.n_layers = args.models.traj.n_layers
        n_heads = args.models.traj.n_heads
        dropout = args.models.traj.dropout

        dec_dims = args.models.traj.decoder_att_dims
        ffn_dims = args.models.traj.decoder_ffn_dims

        self.anchor_step_self_att_layers = nn.ModuleList()
        self.anchor_step_self_att_norm_layers = nn.ModuleList() 
        self.anchor_step_self_att_ffn_layers = nn.ModuleList()
        for l in range(self.n_layers):
            self.anchor_step_self_att_layers.append(MulitiHeadAttention(dec_dims[0], dec_dims[1], n_heads, dropout=dropout))
            self.anchor_step_self_att_norm_layers.append(LayerNorm4D(dec_dims[0]))
            self.anchor_step_self_att_ffn_layers.append(FFN(ffn_dims[0], ffn_dims[1], dropout=dropout))

        self.anchor_step_history_cross_att_layers = nn.ModuleList()
        self.anchor_step_history_cross_att_norm_layers = nn.ModuleList() 
        self.anchor_step_history_cross_att_ffn_layers = nn.ModuleList()
        for l in range(self.n_layers):
            self.anchor_step_history_cross_att_layers.append(MulitiHeadAttention(dec_dims[0], dec_dims[1], n_heads, dropout=dropout))
            self.anchor_step_history_cross_att_norm_layers.append(LayerNorm4D(dec_dims[0]))
            self.anchor_step_history_cross_att_ffn_layers.append(FFN(ffn_dims[0], ffn_dims[1], dropout=dropout))

      
    def forward(self, anchor_step_q, history_kv, self_attn_mask=None, cross_attn_mask=None):
    
        nb, nd, np, ncell = anchor_step_q.size() # ncell=108 for training and ncell=9 for inference
        # print(anchor_step_q.size())
        anchor_step_q = anchor_step_q.permute(0,2,1,3).reshape(nb*np,nd,1,ncell)
        for l in range(self.n_layers):
            attn_anchor_step_q = self.anchor_step_self_att_layers[l](anchor_step_q, anchor_step_q, anchor_step_q, mask=self_attn_mask)
            anchor_step_q = attn_anchor_step_q + anchor_step_q
            anchor_step_q = self.anchor_step_self_att_norm_layers[l](anchor_step_q)
            anchor_step_q = self.anchor_step_self_att_ffn_layers[l](anchor_step_q)
        anchor_step_q = anchor_step_q.reshape(nb,np,nd,ncell).permute(0,2,1,3)

        nb, nd, np, nf = history_kv.size() # nf=19 for training and nf=8,...,19 for inference
        history_kv = history_kv.permute(0,2,1,3).reshape(nb*np,nd,1,nf)
        # print(anchor_step_q.size())
        anchor_step_q = anchor_step_q.permute(0,2,1,3).reshape(nb*np,nd,1,ncell)
        for l in range(self.n_layers):     
            attn_anchor_step_q = self.anchor_step_history_cross_att_layers[l](anchor_step_q, history_kv, history_kv, mask=cross_attn_mask)
            anchor_step_q = attn_anchor_step_q + anchor_step_q
            anchor_step_q = self.anchor_step_history_cross_att_norm_layers[l](anchor_step_q)
            anchor_step_q = self.anchor_step_history_cross_att_ffn_layers[l](anchor_step_q)
        anchor_step_q = anchor_step_q.reshape(nb,np,nd,ncell).permute(0,2,1,3)

        return anchor_step_q