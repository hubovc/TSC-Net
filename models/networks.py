import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os
import time

from models.layers import MulitiHeadAttention, FFN, MLP, Backbone, LayerNorm4D
from models.modules import *
from models.dataset import SceneDataLoader, SceneDataset

from models.utils import *


class TSCNet(nn.Module):
    def __init__(self, args):
        super(TSCNet, self).__init__()

        self.args = args

        self.obs = args.obs_length
        self.future = args.future_length
        self.clip_length = args.clip_length

        self.n_sample = args.num_sample_cvae * args.num_sample_cell

        self.feature_builder = FeatureBuilding(args)
        self.goal_cvae = GoalCVAE(args)
        self.traj_net = TrajNet(args)

    def forward(self, batch_data, epoch):
        if epoch <= self.args.max_epochs:
            loss, recon_conf_loss, recon_offset_loss, kld_loss, traj_conf_loss, traj_offset_loss, goal_fde, min_ade, min_fde = self.train_by_gt(batch_data)

        return loss, recon_conf_loss, recon_offset_loss, kld_loss, traj_conf_loss, traj_offset_loss, goal_fde, min_ade, min_fde

        
    def train_by_gt(self, batch_data):
        spatial_mask, temporal_mask = self.feature_builder.build_spatial_n_temporal_mask(batch_data['num_ped'])        

        # build observation feature     
        cell_attn, anchor_step_feat, anchor_goal_feat, cell_goal_feat = self.feature_builder.build_all_feat_gt(batch_data, spatial_mask, temporal_mask)

        cell_attn_obs = cell_attn[:,:,:,:self.obs]
        anchor_goal_attn = self.feature_builder.build_anchor_goal_attn(anchor_goal_feat, cell_attn_obs)

        # cvae for goal prediction / sampling
        anchor_goal_gt = batch_data['anchor_goal_gt']
        anchor_goal_gtmask = batch_data['anchor_goal_gtmask']
        anchor_goal_recon, mu, log_var = self.goal_cvae(anchor_goal_attn, anchor_goal_gt)

        recon_conf_loss, recon_offset_loss, kld_loss = self.goal_cvae.loss_function(anchor_goal_gt, anchor_goal_recon, anchor_goal_gtmask, mu, log_var)

        # traj net
        nb, nd, np, nf_future, ncell = anchor_step_feat.size()
        nb, nd, np, nf_obs = cell_attn.size()

        anchor_step_feat = anchor_step_feat.reshape(nb, nd, np, nf_future*ncell)
        self_attn_mask, cross_attn_mask = self.feature_builder.build_traj_net_training_mask()
        anchor_step_pred = self.traj_net(anchor_step_feat, cell_attn[:,:,:,:-1], cell_goal_feat, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        anchor_step_pred = anchor_step_pred.reshape(nb, 3, np, nf_future, ncell)        

        anchor_step_gt = batch_data['anchor_step_gt'][:,:,:,self.obs:,:]
        anchor_step_gtmask = batch_data['anchor_step_gtmask'][:,:,:,self.obs:,:]        
        traj_conf_loss, traj_offset_loss = self.traj_net.loss_function(anchor_step_pred, anchor_step_gt, anchor_step_gtmask )
  
        loss = recon_conf_loss + recon_offset_loss + self.goal_cvae.kld_weight * kld_loss + traj_conf_loss + traj_offset_loss


        goal_pred_shift, goal_gt_shift = self.feature_builder.cnovert_anchor_goal_pred_to_traj_training(anchor_goal_recon, batch_data)
        traj_pred_shift, traj_gt_shift = self.feature_builder.cnovert_anchor_step_pred_to_traj_training(anchor_step_pred, batch_data)
        goal_fde, min_ade, min_fde = self.statistic(goal_pred_shift, goal_gt_shift, traj_gt_shift, traj_pred_shift)

        return loss, recon_conf_loss, recon_offset_loss, self.goal_cvae.kld_weight * kld_loss, traj_conf_loss, traj_offset_loss, goal_fde, min_ade, min_fde

    
    def inference(self, batch_data):

        self.feature_builder.n_sample_cvae = 5
        self.feature_builder.n_sample_cell = 4
        self.goal_cvae.n_sample_cvae = 5
        self.goal_cvae.n_sample_cell = 4

        spatial_mask, _ = self.feature_builder.build_spatial_n_temporal_mask(batch_data['num_ped'])  

        cell_feat_obs, cell_attn_obs, anchor_step_feat, anchor_goal_feat, _, traj_feat_ref = self.feature_builder.build_all_feat_pred(batch_data, spatial_mask, False, self.n_sample)
        anchor_goal_attn = self.feature_builder.build_anchor_goal_attn(anchor_goal_feat, cell_attn_obs)
        
        # cvae for goal prediction / sampling
        # anchor_goal_gt = batch_data['anchor_goal_gt']
        # anchor_goal_gtmask = batch_data['anchor_goal_gtmask']

        anchor_goal_pred = self.goal_cvae.sample(anchor_goal_attn)

        goal_pred_shift, cell_goal_feat = self.feature_builder.build_cell_goal_from_pred(batch_data, anchor_goal_pred, traj_feat_ref)

        # traj net for complete the full tarjectory
        dict_update = self.feature_builder.build_dict_update(batch_data, cell_feat_obs, cell_attn_obs, self.n_sample )
        anchor_step_pred = []
        # anchor_step_gt = batch_data['anchor_step_gt'][:,:,:,self.obs:]
        # anchor_step_gtmask = batch_data['anchor_step_gtmask'][:,:,:,self.obs:]
        for f in range(self.obs,self.clip_length):
            anchor_step_pred_cur = self.traj_net(anchor_step_feat, dict_update['cell_attn_history'], cell_goal_feat)
            anchor_step_pred.append(anchor_step_pred_cur)
            anchor_step_feat = self.feature_builder.update_feat_n_dict(dict_update, anchor_step_pred_cur, batch_data, traj_feat_ref, spatial_mask )
        
        goal_pred_shift = goal_pred_shift[:,:2,:,:]
        traj_pred_shift = dict_update['traj_raw_shift_history'][:,:2,:,self.obs:]
        # goal_gt_shift = batch_data['traj_raw_shift'][:,:2,:,-1:].repeat(self.n_sample,1,1,1)
        traj_gt_shift = batch_data['traj_raw_shift'][:,:2].repeat(self.n_sample,1,1,1)
        traj_xyt_ref = batch_data['traj_raw_ref'][:,:2]

        goal_fde, min_ade, min_fde = self.statistic(goal_pred_shift, traj_gt_shift[:,:,:,-1:], traj_gt_shift[:,:,:,self.obs:], traj_pred_shift)

        return goal_pred_shift, traj_pred_shift, traj_gt_shift, traj_xyt_ref, goal_fde, min_ade, min_fde

    def inference_hybrid_record(self, batch_data):
        m_n_list = [[20,1],[10,2],[5,4],[4,5],[2,10],[1,20]]

        spatial_mask, _ = self.feature_builder.build_spatial_n_temporal_mask(batch_data['num_ped'])  

        cell_feat_obs, cell_attn_obs, anchor_step_feat, anchor_goal_feat, _, traj_feat_ref = self.feature_builder.build_all_feat_pred(batch_data, spatial_mask, False, self.n_sample)
        anchor_goal_attn = self.feature_builder.build_anchor_goal_attn(anchor_goal_feat, cell_attn_obs)
        

        goal_fde_record_all = []
        for x in range(2) :
            goal_fde_record = []

            for (m,n) in m_n_list:
                self.feature_builder.n_sample_cvae = m
                self.feature_builder.n_sample_cell = n
                self.goal_cvae.n_sample_cvae = m
                self.goal_cvae.n_sample_cell = n

                anchor_goal_pred = self.goal_cvae.sample(anchor_goal_attn)
                goal_pred_shift, cell_goal_feat = self.feature_builder.build_cell_goal_from_pred(batch_data, anchor_goal_pred, traj_feat_ref)
                goal_gt_shift = batch_data['traj_raw_shift'][:,:2,:,-1:].repeat(self.n_sample,1,1,1)
                # print(goal_pred_shift.size(), goal_gt_shift.size())
                goal_dist = euclidean_distance(goal_pred_shift[:,:2,:,:], goal_gt_shift)[:,0,:,:] # nb x np x 1
                goal_fde = goal_dist.min(dim=0)[0]

                # print(goal_fde.size())#, goal_fde.size(), anchor_goal_attn.size()) #np
                goal_fde_record.append(goal_fde)
            # tmp = input()

            goal_fde_record = torch.stack(goal_fde_record, dim=0) # 6 x np
            min_goal_fde_mn_idx_one_round = goal_fde_record.argmin(dim=0)
            goal_fde_record_all.append(min_goal_fde_mn_idx_one_round)

        goal_fde_record_all = torch.cat(goal_fde_record_all, dim=1)        

        goal_fde_count = [ [ (goal_fde_record_all[i] == j).sum().item() for j in range(6) ] for i in range(goal_fde_record_all.size(0))]    
        goal_fde_count = torch.IntTensor(goal_fde_count)

        min_goal_fde_mn_idx = goal_fde_count.argmax(dim=1)
        # print(goal_fde_count, min_goal_fde_mn_idx)
        # tmp = input()
        traj_gt_shift = batch_data['traj_raw_shift'][:,:2]

        return traj_gt_shift, min_goal_fde_mn_idx

    def inference_hybrid(self, batch_data, min_goal_fde_mn_idx):
        m_n_list = [[20,1],[10,2],[5,4],[4,5],[2,10],[1,20]]
        
        spatial_mask, _ = self.feature_builder.build_spatial_n_temporal_mask(batch_data['num_ped'])  

        cell_feat_obs, cell_attn_obs, anchor_step_feat, anchor_goal_feat, _, traj_feat_ref = self.feature_builder.build_all_feat_pred(batch_data, spatial_mask, False, self.n_sample)
        anchor_goal_attn = self.feature_builder.build_anchor_goal_attn(anchor_goal_feat, cell_attn_obs)
        
        goal_pred_shift_all_ped = []
        cell_goal_feat_all_ped = []
        for p in range(anchor_goal_attn.size(2)):
            m, n = m_n_list[min_goal_fde_mn_idx[p]]
            anchor_goal_pred_one_ped = self.goal_cvae.sample_one_ped(anchor_goal_attn[:,:,p:p+1,:], m)
            goal_pred_shift_one_ped, cell_goal_feat_one_ped = self.feature_builder.build_cell_goal_from_pred_one_ped(batch_data, anchor_goal_pred_one_ped, traj_feat_ref, p, n)

            goal_pred_shift_all_ped.append(goal_pred_shift_one_ped)
            cell_goal_feat_all_ped.append(cell_goal_feat_one_ped)    

        goal_pred_shift = torch.cat(goal_pred_shift_all_ped, dim=2)    
        cell_goal_feat = torch.cat(cell_goal_feat_all_ped, dim=2)    

        # traj net for complete the full tarjectory
        dict_update = self.feature_builder.build_dict_update(batch_data, cell_feat_obs, cell_attn_obs, self.n_sample )
        anchor_step_pred = []
        # anchor_step_gt = batch_data['anchor_step_gt'][:,:,:,self.obs:]
        # anchor_step_gtmask = batch_data['anchor_step_gtmask'][:,:,:,self.obs:]
        for f in range(self.obs,self.clip_length):
            anchor_step_pred_cur = self.traj_net(anchor_step_feat, dict_update['cell_attn_history'], cell_goal_feat)
            anchor_step_pred.append(anchor_step_pred_cur)
            anchor_step_feat = self.feature_builder.update_feat_n_dict(dict_update, anchor_step_pred_cur, batch_data, traj_feat_ref, spatial_mask )
        
        goal_pred_shift = goal_pred_shift[:,:2,:,:]
        traj_pred_shift = dict_update['traj_raw_shift_history'][:,:2,:,self.obs:]
        # goal_gt_shift = batch_data['traj_raw_shift'][:,:2,:,-1:].repeat(self.n_sample,1,1,1)
        traj_gt_shift = batch_data['traj_raw_shift'][:,:2].repeat(self.n_sample,1,1,1)
        traj_xyt_ref = batch_data['traj_raw_ref'][:,:2]

        goal_fde, min_ade, min_fde = self.statistic(goal_pred_shift, traj_gt_shift[:,:,:,-1:], traj_gt_shift[:,:,:,self.obs:], traj_pred_shift)

        return goal_pred_shift, traj_pred_shift, traj_gt_shift, traj_xyt_ref, goal_fde, min_ade, min_fde, min_goal_fde_mn_idx


    def statistic(self, goal_pred_shift, goal_gt_shift, traj_gt_shift, traj_pred_shift):
        # print(traj_gt_shift.size(), traj_pred_shift.size())
        traj_dist = euclidean_distance(traj_gt_shift, traj_pred_shift)[:,0,:,:] # nb x np x nf
        ade = traj_dist.mean(dim=2) # nb x np
        fde = traj_dist[:,:,-1] # nb x np

        min_ade = ade.min(dim=0)[0] # np
        min_fde = fde.min(dim=0)[0] # np

        min_ade = min_ade.mean()
        min_fde = min_fde.mean()

        # print(goal_pred_shift.size(), goal_gt_shift.size())
        goal_dist = euclidean_distance(goal_pred_shift, goal_gt_shift)[:,0,:,:] # nb x np x 1
        goal_fde = goal_dist.min(dim=0)[0]
        goal_fde = goal_fde.mean()

        return goal_fde, min_ade, min_fde



class GoalCVAE(nn.Module):
    def __init__(self, args):
        super(GoalCVAE, self).__init__()
        self.args = args

        self.gt_emb = MLP(args.models.goal.gt_emb_dims)
        self.cvae_encoder = MLP(args.models.goal.encoder_dims)
        self.cvae_decoder = MLP(args.models.goal.decoder_dims)
        self.latent_dim = args.models.goal.latent_dim

        self.kld_weight = args.models.goal.kld_weight
        self.recon_conf_weight = args.models.goal.recon_conf_weight
        self.recon_offset_weight = args.models.goal.recon_offset_weight
    
        self.n_sample_cvae = args.num_sample_cvae
        self.n_sample_cell = args.num_sample_cell

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu
    
    def forward(self, anchor_goal_condition, anchor_goal_gt):

        anchor_goal_gtemb = self.gt_emb(anchor_goal_gt)
        enc_input = torch.cat((anchor_goal_gtemb,anchor_goal_condition), dim=1)
        latent = self.cvae_encoder(enc_input)

        nb, nd, np, ncell = latent.size()
        latent = latent.reshape(nb,2,int(nd/2),np,ncell)
        mu, log_var = latent[:,0], latent[:,1]
        z = self.reparameterize(mu, log_var) # 1 x d(64) x N x ncell(225)

        dec_input = torch.cat((z,anchor_goal_condition), dim=1)
        anchor_goal_recon = self.cvae_decoder(dec_input)

        anchor_goal_recon_conf = torch.sigmoid(anchor_goal_recon[:,0:1])
        anchor_goal_recon_offset = anchor_goal_recon[:,1:]
        anchor_goal_recon_offset[anchor_goal_recon_offset<-40.] = -40.
        anchor_goal_recon_offset[anchor_goal_recon_offset> 40.] =  40.

        anchor_goal_recon = torch.cat((anchor_goal_recon_conf, anchor_goal_recon_offset), dim=1)
        # for p in range(anchor_goal_recon.size(2)):
        #     print(anchor_goal_recon[0,1,p].resize(15,15))
        #     print(anchor_goal_recon[0,2,p].resize(15,15))
        #     print(anchor_goal_recon[0,0,p].resize(15,15))
        #     tmp = input()
        
        return anchor_goal_recon, mu, log_var

    def sample(self, anchor_goal_condition):
        nb, nd, np_all, ncell = anchor_goal_condition.size()
        anchor_goal_condition = anchor_goal_condition.repeat(self.n_sample_cvae,1,1,1)
        
        z = torch.randn(self.n_sample_cvae, self.latent_dim, np_all, ncell).cuda() #.repeat(1,1,np_all,1)

        dec_input = torch.cat((z,anchor_goal_condition), dim=1)
        anchor_goal_pred = self.cvae_decoder(dec_input)

        anchor_goal_pred_conf = torch.sigmoid(anchor_goal_pred[:,0:1])
        anchor_goal_pred_offset = anchor_goal_pred[:,1:]
        anchor_goal_pred_offset[anchor_goal_pred_offset<-40.] = -40.
        anchor_goal_pred_offset[anchor_goal_pred_offset> 40.] =  40.

        anchor_goal_pred = torch.cat((anchor_goal_pred_conf, anchor_goal_pred_offset), dim=1)

        return anchor_goal_pred

    def sample_one_ped(self, anchor_goal_condition, n_sample_cvae):
        nb, nd, np_all, ncell = anchor_goal_condition.size()
        anchor_goal_condition = anchor_goal_condition.repeat(n_sample_cvae,1,1,1)
        
        z = torch.randn(n_sample_cvae, self.latent_dim, np_all, ncell).cuda() #.repeat(1,1,np_all,1)

        dec_input = torch.cat((z,anchor_goal_condition), dim=1)
        anchor_goal_pred = self.cvae_decoder(dec_input)

        anchor_goal_pred_conf = torch.sigmoid(anchor_goal_pred[:,0:1])
        anchor_goal_pred_offset = anchor_goal_pred[:,1:]
        # anchor_goal_pred_offset[anchor_goal_pred_offset<-30.] = -30.
        # anchor_goal_pred_offset[anchor_goal_pred_offset> 30.] =  30.

        anchor_goal_pred_offset_alter = torch.rand_like(anchor_goal_pred_offset) * 60 - 30
        anchor_goal_pred_offset[anchor_goal_pred_offset<-30.] = anchor_goal_pred_offset_alter[anchor_goal_pred_offset<-30.]
        anchor_goal_pred_offset[anchor_goal_pred_offset> 30.] = anchor_goal_pred_offset_alter[anchor_goal_pred_offset> 30.]

        anchor_goal_pred = torch.cat((anchor_goal_pred_conf, anchor_goal_pred_offset), dim=1)
        # anchor_goal_pred = torch.cat((anchor_goal_pred_conf, anchor_goal_pred_offset_alter), dim=1)
        
        return anchor_goal_pred
        
    def loss_function(self, anchor_goal_gt, anchor_goal_recon, anchor_goal_gtmask, mu, log_var):
        # recon_loss = F.mse_loss(anchor_goal_gt*anchor_goal_gtmask, anchor_goal_recon*anchor_goal_gtmask)
        recon_conf_loss = torch.nn.SmoothL1Loss()(anchor_goal_gt[:,0:1]*anchor_goal_gtmask[:,0:1], anchor_goal_recon[:,0:1]*anchor_goal_gtmask[:,0:1])
        recon_offset_loss = torch.nn.SmoothL1Loss()(anchor_goal_gt[:,1:]*anchor_goal_gtmask[:,1:], anchor_goal_recon[:,1:]*anchor_goal_gtmask[:,1:])
        
        kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss = kld_loss.sum(dim=1).mean()

        return recon_conf_loss, recon_offset_loss, kld_loss

class TrajNet(nn.Module):
    def __init__(self, args):
        super(TrajNet, self).__init__()

        self.step_decoder = StepDecoder(args)
        self.prediction_layer = MLP(args.models.traj.prediction_dims)

        self.conf_weight = args.models.traj.conf_weight
        self.offset_weight = args.models.traj.offset_weight

    def forward(self, anchor_step_feat_query, cell_attn_history_kv, cell_goal_feat, self_attn_mask=None, cross_attn_mask=None):
        anchor_step_attn = self.step_decoder(anchor_step_feat_query, cell_attn_history_kv, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)

        anchor_step_attn = torch.cat((anchor_step_attn, cell_goal_feat.repeat(1,1,1,anchor_step_attn.size(3))), dim=1)
        anchor_step_pred = self.prediction_layer(anchor_step_attn)

        anchor_step_pred_conf = torch.sigmoid(anchor_step_pred[:,0:1])
        anchor_step_pred_offset = anchor_step_pred[:,1:]

        anchor_step_pred = torch.cat((anchor_step_pred_conf, anchor_step_pred_offset), dim=1)

        return anchor_step_pred

    def loss_function(self, anchor_step_pred, anchor_step_gt, anchor_step_gtmask):
        conf_loss   = torch.nn.SmoothL1Loss()(anchor_step_gt[:,0:1]*anchor_step_gtmask[:,0:1], anchor_step_pred[:,0:1]*anchor_step_gtmask[:,0:1])
        offset_loss = torch.nn.SmoothL1Loss()(anchor_step_gt[:,1:]*anchor_step_gtmask[:,1:], anchor_step_pred[:,1:]*anchor_step_gtmask[:,1:])

        return conf_loss, offset_loss





        # pred_max_conf_idx = pred_cells[0,0,:,:].argmax(dim=1).reshape(-1)
        # gt_overlap = gt_cells[0,0,torch.LongTensor(range(gt_cells.size(2))),pred_max_conf_idx]
        # print(gt_overlap.size())

    