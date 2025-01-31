import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os

class Attention(nn.Module):
    def __init__(self, dropout):
        super(Attention, self).__init__()    
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self,q,k,v,mask=None):
        (n_b,n_q,n_dq) = q.data.size()
        (n_b,n_dk,n_k) = k.data.size()
        (n_b,n_v,n_dv) = v.data.size()

        # print(q.size(),k.size())
        att_mat = torch.matmul(q,k)
        if mask is not None:
            att_mat = att_mat.masked_fill(mask == 0, -1e9)
        att_mat = F.softmax( att_mat, dim=-1 ) / (n_dk**0.5)             # (n_b,n_q,n_k) 
        if self.dropout > 0:
            att_mat = self.dropout_layer(att_mat)
        att_q = torch.matmul(att_mat,v).permute(0,2,1).reshape(n_b,n_dv,n_q) # (n_b,n_dv,n_q)        

        return att_q

class MulitiHeadAttention(nn.Module):
    def __init__(self, in_dim, inter_dim, n_head, dropout=0.):
        super(MulitiHeadAttention, self).__init__()
        if isinstance(in_dim, tuple):
            self.in_dim_q = in_dim[0]
            self.in_dim_k = in_dim[1]
            self.in_dim_v = in_dim[2]
        else:
            self.in_dim_q = in_dim
            self.in_dim_k = in_dim
            self.in_dim_v = in_dim

        if isinstance(inter_dim, tuple):
            self.inter_dim_qk = inter_dim[0]
            self.inter_dim_v  = inter_dim[1]
        else:
            self.inter_dim_qk = inter_dim
            self.inter_dim_v  = inter_dim

        self.out_dim    = self.in_dim_q

        self.n_head     = n_head
        self.dropout    = dropout
        self.one_dim_qk = int(self.inter_dim_qk/self.n_head)
        self.one_dim_v  = int(self.inter_dim_v/self.n_head)

        if self.inter_dim_qk % self.n_head != 0 or self.inter_dim_v % self.n_head != 0:
            print("inter_dim_qk(%d) / n_head(%d) should be an integer."%(self.inter_dim_qk,self.n_head))
            print("inter_dim_v(%d)  / n_head(%d) should be an integer."%(self.inter_dim_v,self.n_head))
            tmp = input()

        self.create_network()

    def create_network(self):
        self.w_q = nn.Conv2d(self.in_dim_q, self.inter_dim_qk, 1, 1, 0)
        self.w_k = nn.Conv2d(self.in_dim_k, self.inter_dim_qk, 1, 1, 0)
        self.w_v = nn.Conv2d(self.in_dim_v, self.inter_dim_v,  1, 1, 0)

        self.w_out = nn.Conv2d(self.inter_dim_v, self.out_dim, 1, 1, 0)
        self.attention = Attention(self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)

        torch.nn.init.xavier_uniform_(self.w_q.weight)
        torch.nn.init.xavier_uniform_(self.w_k.weight)
        torch.nn.init.xavier_uniform_(self.w_v.weight)
        torch.nn.init.xavier_uniform_(self.w_out.weight)

    def forward(self,q_in,k_in,v_in,mask=None):
        (n_b,n_dq,h_q,w_q) = q_in.data.size()
        (n_b,n_dk,h_k,w_k) = k_in.data.size()
        (n_b,n_dk,h_v,w_v) = v_in.data.size()
        n_h   = self.n_head
        n_dqk = self.one_dim_qk
        n_dv  = self.one_dim_v
        if h_k*w_k % h_v*w_v != 0:
            print("Samples in Key (%d) and Value(%d) should be Equal."%(h_k*w_k,h_v*w_v))
            tmp = raw_input()

        q = self.w_q(q_in).reshape(n_b*n_h,n_dqk,h_q*w_q).permute(0,2,1)                                 # (n_b*n_h,h_q*w_q,nDI)
        k = self.w_k(k_in).reshape(n_b*n_h,n_dqk,h_k*w_k)                                                # (n_b*n_h,nDI,h_k*w_k)
        v = self.w_v(v_in).reshape(n_b*n_h,n_dv ,h_v*w_v).permute(0,2,1)                                 # (n_b*n_h,h_v*w_v,nDI)

        # print(q_in.size(), q.size(), k.size(), v.size())

        if mask is not None:
            n_pq, n_pkv = mask.size()
            mask  = mask.reshape(1,n_pq,n_pkv).repeat(n_b*n_h,1,1).cuda()

        att_q = self.attention(q,k,v,mask).reshape(n_b,n_h,n_dv,h_q*w_q).reshape(n_b,n_h*n_dv,h_q,w_q)
        att_q = self.w_out(att_q).reshape(n_b,self.out_dim,h_q,w_q)

        if self.dropout > 0:
            att_q = self.dropout_layer(att_q)
        
        return att_q


class LayerNorm4D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm4D, self).__init__()
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.norm.weight)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        # x size: nb x nd x np x nf
        norm_x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)  
        return norm_x


class FFN(nn.Module):
    def __init__(self, in_dim, inter_dim, dropout=0.):
        super(FFN, self).__init__()
        out_dim  = in_dim
        self.dropout = dropout

        self.ffn          = nn.Sequential(nn.Conv2d(in_dim,inter_dim,1,1,0),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(inter_dim,out_dim,1,1,0))

        self.dropout_layer = nn.Dropout(self.dropout)
        self.ffn_norm      = LayerNorm4D(out_dim)

    def forward(self,x):
        (nb,nd,np,nf) = x.data.size()
        ffn_x = self.ffn(x)
        if self.dropout > 0:
            ffn_x = self.dropout_layer(ffn_x)
        
        ffn_x = ffn_x + x
        ffn_x = self.ffn_norm(ffn_x)  

        return ffn_x


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for l in range(len(dims)-1):
            if l+1 == len(dims)-1:
                self.layers.append(nn.Conv2d(dims[l],dims[l+1],1,1,0))
            else:
                self.layers.append(nn.Sequential(nn.Conv2d(dims[l],dims[l+1],1,1,0), nn.LeakyReLU(0.1, inplace=True)))

    def positional_encoding(self, D, T, n_split=1 ):
        D = int(D/n_split)
        pe = torch.zeros(D,T).cuda()
        for t in range(T):
            ft = torch.tensor(t).float()
            for d in range(D):
                fd = torch.tensor(d).float()
                if d % 2 == 0:
                    pe[d,t] = torch.sin( t / torch.pow( 10000, fd / D ) )
                else:
                    pe[d,t] = torch.cos( t / torch.pow( 10000, fd / D ) )
        return pe

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        self.scene_goal_backbone = nn.ModuleList()
        self.scene_step_backbone = nn.ModuleList()

        self.conv_list = []
        self.pool_list = []
        channels = args.models.feat.scene_dims
        # w = h = args.models.goal.scene_size
        for i in range(len(channels)-1):
            k = (5,5) if i <= 2 else (3,3)
            s = (1,1) if i != 2 else (5,5)
            if i < 2:                
                p = (2,2)
            elif i == 2:
                p = (0,0)
            else:
                p = (1,1)

            if i == len(channels)-2:
                self.conv_list.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=k, stride=s, padding=p))
            else:
                self.conv_list.append(nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=k, stride=s, padding=p),
                    nn.LeakyReLU(0.1, inplace=True)))

        for i in range(2):
            self.pool_list.append(nn.MaxPool2d(2,stride=2))

        for i in range(len(self.conv_list)):
            self.scene_goal_backbone.append(self.conv_list[i])
            self.scene_step_backbone .append(self.conv_list[i])
            if i == 3:
                self.scene_goal_backbone.append(self.pool_list[0])
            if i == 5:
                self.scene_goal_backbone.append(self.pool_list[1])

        self.local_to_one = nn.Conv2d(channels[-1], channels[-1], kernel_size=(3,3), stride=(3,3), padding=(0,0))
        self.norm = LayerNorm4D(channels[-1])
        

    def scene_goal_emb(self, x):
        # x is all scene semantic maps, shape: n_batch x c x h x w
        nb, nc, nh, nw = x.size()
        
        for layer in self.scene_goal_backbone:
            x = layer(x)

        # x = self.norm(x)

        return x

    def scene_step_emb_to_3x3(self, x):
        # x is all scene semantic maps, shape: n_batch x c x h x w
        np, nf, nc, nh, nw = x.size()
        x = x.reshape(np*nf, nc, nh, nw)
        for layer in self.scene_step_backbone:
            x = layer(x)

        # x = self.norm(x)
        nd, nh, nw = x.size()[1:]
        x = x.reshape(np, nf, nd, nh, nw)
        x = x[:,:,:,3:6,3:6] # select the center position

        return x

    def scene_step_emb_to_1x1(self, x):
        np, nf, nc, nh, nw = x.size()
        x = x.reshape(np*nf, nc, nh, nw)
        x = self.local_to_one(x)

        nd, nh, nw = x.size()[1:]
        x = x.reshape(1, np, nf, nd).permute(0,3,1,2)

        return x

    def forward(self,x):
        return x