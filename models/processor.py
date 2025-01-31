import pandas as pd
import numpy as np
import torch
import pickle
import os
import time
from tqdm import tqdm

from models.networks import TSCNet
from models.dataset import SceneDataLoader, SceneDataset

def train_tsc_net(args, init_epoch ):
    weight_dir = os.path.join(args.weight_dir,args.dataset_name)
    result_dir = os.path.join(args.result_dir,args.dataset_name,'train')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = int(time.time())
    torch.manual_seed(seed)
    if args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        torch.cuda.manual_seed(seed)

    if init_epoch > 0:
        weight_file = os.path.join(weight_dir,'TSC_Net_{}_Ep_{}'.format(args.dataset_name,init_epoch))
        model = torch.load(weight_file)
    else:
        model = TSCNet(args)

    if args.use_cuda:
        model = model.cuda()  

    optimizer = torch.optim.Adam(model.parameters(), lr=args.models.learning_rate)

    train_dataset = SceneDataset(args, 'train')
    train_loader = SceneDataLoader(train_dataset, batch_size=1)

    test_dataset = SceneDataset(args, 'test')
    test_loader = SceneDataLoader(test_dataset, batch_size=1)

    model.seen = init_epoch * train_dataset.num_ped_samples

    for epoch in range(init_epoch, args.max_epochs): 
        
        train_loader.dataset.build_train_batch_list()

        print('Training Epoch %d, Train Set Size = %d' % (epoch+1,train_dataset.num_scene_samples))
        train_tsc_net_epoch(args, model, optimizer, train_loader, epoch)

        weight_file = os.path.join(weight_dir,'TSC_Net_{}_Ep_{}'.format(args.dataset_name,epoch+1))
        result_file = os.path.join(result_dir,'TSC_Net_{}_Ep_{}.res'.format(args.dataset_name,epoch+1))

        if (epoch+1) % args.save_interval == 0:
            torch.save(model, weight_file)

        test_tsc_net(args,epoch+1)
        # print('Testing Epoch %d, Test Set Size = %d' % (epoch+1,test_dataset.num_scene_samples))
        # all_outputs = test_goal_network_epoch(args, model, test_loader)
        # with open(result_file, "wb") as f:
        #     pickle.dump(all_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)

        # min_avg_fde, min_min_fde, min_max_fde = eval_goal_epoch(args, all_outputs)
        # eval_file = open('goal_network_performance.txt', 'a')
        # eval_file.write('%d %0.3f %0.3f %0.3f\n' % (epoch, min_avg_fde, min_min_fde, min_max_fde))
        # eval_file.close()
        # print('    min_avg_fde=%0.3f, min_min_fde=%0.3f, min_max_fde=%0.3f' % (min_avg_fde, min_min_fde, min_max_fde))

def train_tsc_net_epoch(args, model, optimizer, train_loader, epoch):
    model.train()

    for batch_idx, batch_data in enumerate(train_loader):       
        model.seen = model.seen + len(batch_data['num_ped'])

        optimizer.zero_grad()
        loss, goal_conf_loss, goal_offset_loss, kld_loss, traj_conf_loss, traj_offset_loss, goal_fde, min_ade, min_fde = model(batch_data, epoch)

        loss.backward()
        optimizer.step()

        # if batch_idx % 10 == 0:
        print('%6d: ade=%0.2f, fde=%0.2f, goal_conf_loss=%0.3f, goal_offset_loss=%0.3f, kld_loss=%0.3f, traj_conf_loss = %0.3f, traj_offset_loss=%0.3f, loss=%0.3f' % 
            (model.seen, min_ade.item(), goal_fde.item(), goal_conf_loss.item(), goal_offset_loss.item(), kld_loss.item(), traj_conf_loss.item(), traj_offset_loss.item(), loss.item()))


def test_tsc_net(args, target_epoch=None):
    weight_dir = os.path.join(args.weight_dir,args.dataset_name)
    result_dir = os.path.join(args.result_dir,args.dataset_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = int(time.time())
    torch.manual_seed(seed)

    if args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        torch.cuda.manual_seed(seed)

    train_dataset = SceneDataset(args, 'train')
    train_loader = SceneDataLoader(train_dataset, batch_size=1)

    test_dataset = SceneDataset(args, 'test')
    test_loader = SceneDataLoader(test_dataset, batch_size=1)

    for epoch in range(args.max_epochs):

        if target_epoch is not None and epoch != target_epoch:
            continue

        weight_file = os.path.join(weight_dir,'TSC_Net_{}_Ep_{}'.format(args.dataset_name,epoch))
        result_file = os.path.join(result_dir,'TSC_Net_{}_Ep_{}.res'.format(args.dataset_name,epoch))

        if not os.path.exists(weight_file):
            continue

        print('Loading weight file from %s.'%weight_file)
        model = torch.load(weight_file)
        if args.use_cuda:
            model = model.cuda()

        all_outputs = test_tsc_net_epoch(args, model, test_loader, epoch)
        goal_fde, min_ade, min_fde = eval_tsc_net_epoch(args, epoch, model, all_outputs)

        with open(result_file, "wb") as f:
            pickle.dump(all_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)

        eval_file = open('goal_network_performance.txt', 'a')
        eval_file.write('%d %0.3f %0.3f %0.3f\n' % (epoch, goal_fde, min_ade, min_fde))
        eval_file.close()

def test_tsc_net_epoch(args, model, test_loader, epoch):
    model.eval()
    model.test_seen = 0
    all_goal_pred_shift = []
    all_traj_pred_shift = []
    all_traj_gt_shift = []
    all_traj_gt_ref = []
    num_ped_list = []

    pbar = tqdm(total=len(test_loader.dataset))
    pbar.set_description('Testing epoch %d'%(epoch))
    for batch_idx, batch_data in enumerate(test_loader):    
        model.test_seen = model.test_seen + len(batch_data['num_ped'])

        goal_pred_shift, traj_pred_shift, traj_gt_shift, traj_xyt_ref, goal_fde, min_ade, min_fde = model.inference(batch_data)        

        all_goal_pred_shift.append(goal_pred_shift.detach().cpu())
        all_traj_pred_shift.append(traj_pred_shift.detach().cpu())
        all_traj_gt_shift.append(traj_gt_shift.detach().cpu())
        all_traj_gt_ref.append(traj_xyt_ref.detach().cpu())
        num_ped_list.extend(batch_data['num_ped'])

        pbar.update(len(batch_data['num_ped']))

    pbar.close()

    all_goal_pred_shift = torch.cat(all_goal_pred_shift,dim=2)
    all_traj_pred_shift = torch.cat(all_traj_pred_shift,dim=2)
    all_traj_gt_shift = torch.cat(all_traj_gt_shift,dim=2)
    all_traj_gt_ref = torch.cat(all_traj_gt_ref,dim=2)
    
    return (all_goal_pred_shift, all_traj_pred_shift, all_traj_gt_shift, all_traj_gt_ref, num_ped_list)


def eval_tsc_net(args, model, target_epoch=None):
    all_fde = []
    for epoch in range(init_epoch, args.max_epochs):
        if target_epoch is not None and epoch != target_epoch:
            continue

        result_file = os.path.join(result_dir,'TSC_Net_{}_Ep_{}.res'.format(args.dataset_name,epoch))
        with open(result_file, "rb") as f:
            all_outputs = pickle.load(f)
            
        goal_fde, min_ade, min_fde = eval_tsc_net_epoch(args, epoch, model, all_outputs)
        all_fde.append([epoch, fde])

    np.savetxt('tsc_network_performance.txt', np.array(all_fde), fmt='%0.3f')

def eval_tsc_net_epoch_hybrid(args, epoch, model, all_outputs):
    goal_pred_shift, future_pred_shift, traj_gt_shift, all_traj_gt_ref, num_ped_list, mn_list = all_outputs
    goal_gt_shift = traj_gt_shift[:,:,:,-1:]
    future_gt_shift = traj_gt_shift[:,:,:,args.obs_length:]

    goal_fde, min_ade, min_fde = model.statistic(goal_pred_shift, goal_gt_shift, future_gt_shift, future_pred_shift)    

    print('    epoch-%d: goal_fde=%0.3f, min_ade=%0.3f, min_fde=%0.3f' % (epoch, goal_fde, min_ade, min_fde))

    return goal_fde, min_ade, min_fde

def eval_tsc_net_epoch(args, epoch, model, all_outputs):
    goal_pred_shift, future_pred_shift, traj_gt_shift, all_traj_gt_ref, num_ped_list = all_outputs
    goal_gt_shift = traj_gt_shift[:,:,:,-1:]
    future_gt_shift = traj_gt_shift[:,:,:,args.obs_length:]

    goal_fde, min_ade, min_fde = model.statistic(goal_pred_shift, goal_gt_shift, future_gt_shift, future_pred_shift)    

    print('    epoch-%d: goal_fde=%0.3f, min_ade=%0.3f, min_fde=%0.3f' % (epoch, goal_fde, min_ade, min_fde))

    return goal_fde, min_ade, min_fde
    # tmp = input()

def test_tsc_net_hybrid_record(args, target_epoch=None):
    weight_dir = os.path.join(args.weight_dir,args.dataset_name)
    result_dir = os.path.join(args.result_dir,args.dataset_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = int(time.time())
    torch.manual_seed(seed)

    if args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        torch.cuda.manual_seed(seed)

    # train_dataset = SceneDataset(args, 'train')
    # train_loader = SceneDataLoader(train_dataset, batch_size=1)

    # test_dataset = SceneDataset(args, 'train')
    test_dataset = SceneDataset(args, 'test')
    test_loader = SceneDataLoader(test_dataset, batch_size=1)

    for epoch in range(args.max_epochs):
        if target_epoch is not None and epoch != target_epoch:
            continue

        weight_file = os.path.join(weight_dir,'TSC_Net_{}_Ep_{}'.format(args.dataset_name,epoch))
        # train_data_file = os.path.join()
        # test_data_file = os.path.join()

        # print(weight_file)
        if not os.path.exists(weight_file):
            continue

        model = torch.load(weight_file)
        if args.use_cuda:
            model = model.cuda()

        # all_outputs = test_tsc_net_epoch(args, model, test_loader, epoch)
        all_outputs = test_tsc_net_epoch_hybrid_for_record(args, model, test_loader, epoch, 'testing')

        x = all_outputs[0][0,:2,:,:8].permute(1,0,2).reshape(-1,16)
        y = all_outputs[1]
        # print(x.size(),y.size())
        train_data = torch.cat((x,y.reshape(-1,1)), dim=1)
        with open('data/testing_mn_data.pkl', 'wb') as f:
            pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    test_dataset = SceneDataset(args, 'test', test_train=True)
    test_loader = SceneDataLoader(test_dataset, batch_size=1)

    for epoch in range(args.max_epochs):
        if target_epoch is not None and epoch != target_epoch:
            continue

        weight_file = os.path.join(weight_dir,'TSC_Net_{}_Ep_{}'.format(args.dataset_name,epoch))
        # train_data_file = os.path.join()
        # test_data_file = os.path.join()

        # print(weight_file)
        if not os.path.exists(weight_file):
            continue

        model = torch.load(weight_file)
        if args.use_cuda:
            model = model.cuda()

        # all_outputs = test_tsc_net_epoch(args, model, test_loader, epoch)
        all_outputs = test_tsc_net_epoch_hybrid_for_record(args, model, test_loader, epoch, 'training')

        x = all_outputs[0][0,:2,:,:8].permute(1,0,2).reshape(-1,16)
        y = all_outputs[1]
        # print(x.size(),y.size())
        train_data = torch.cat((x,y.reshape(-1,1)), dim=1)
        with open('data/training_mn_data.pkl', 'wb') as f:
            pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def test_tsc_net_epoch_hybrid_for_record(args, model, test_loader, epoch, phase):
    model.eval()
    model.test_seen = 0

    num_ped_list = []
    all_traj_gt_shift = []
    min_mn_idx = []


    pbar = tqdm(total=len(test_loader.dataset))
    pbar.set_description('Testing epoch %d with different (m,n) on %s set'%(epoch, phase))
    for batch_idx, batch_data in enumerate(test_loader):    
        model.test_seen = model.test_seen + len(batch_data['num_ped'])

        traj_gt_shift, min_goal_fde_mn_idx = model.inference_hybrid_record(batch_data)        

        all_traj_gt_shift.append(traj_gt_shift.detach().cpu())
        min_mn_idx.append(min_goal_fde_mn_idx.detach().cpu())
        num_ped_list.extend(batch_data['num_ped'])


        pbar.update(len(batch_data['num_ped']))

    pbar.close()

    min_mn_idx = torch.cat(min_mn_idx, dim=0)
    all_traj_gt_shift = torch.cat(all_traj_gt_shift,dim=2)
    
    return (all_traj_gt_shift, min_mn_idx)

def test_tsc_net_hybrid(args, target_epoch=None):
    weight_dir = os.path.join(args.weight_dir,args.dataset_name)
    result_dir = os.path.join(args.result_dir,args.dataset_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open('data/mn_idx.pkl', 'rb') as f:
        min_goal_fde_mn_idx = pickle.load(f)

    seed = int(time.time())
    torch.manual_seed(seed)

    if args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        torch.cuda.manual_seed(seed)

    test_dataset = SceneDataset(args, 'test')
    test_loader = SceneDataLoader(test_dataset, batch_size=1)

    for epoch in range(args.max_epochs):
        if target_epoch is not None and epoch != target_epoch:
            continue

        weight_file = os.path.join(weight_dir,'TSC_Net_{}_Ep_{}'.format(args.dataset_name,epoch))
        result_file = os.path.join(result_dir,'TSC_Net_{}_Ep_{}_hybrid.res'.format(args.dataset_name,epoch))
        # train_data_file = os.path.join()
        # test_data_file = os.path.join()

        # print(weight_file)
        if not os.path.exists(weight_file):
            continue

        model = torch.load(weight_file)
        if args.use_cuda:
            model = model.cuda()

        # all_outputs = test_tsc_net_epoch(args, model, test_loader, epoch)
        all_outputs = test_tsc_net_epoch_hybrid(args, model, test_loader, min_goal_fde_mn_idx, epoch)

        with open(result_file, "wb") as f:
            pickle.dump(all_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)

        goal_fde, min_ade, min_fde = eval_tsc_net_epoch_hybrid(args, epoch, model, all_outputs)

        eval_file = open('goal_network_performance.txt', 'a')
        eval_file.write('%d %0.3f %0.3f %0.3f\n' % (epoch, goal_fde, min_ade, min_fde))
        eval_file.close()

def test_tsc_net_epoch_hybrid(args, model, test_loader, min_goal_fde_mn_idx, epoch):
    model.eval()
    model.test_seen = 0
    all_goal_pred_shift = []
    all_traj_pred_shift = []
    all_traj_gt_shift = []
    all_traj_gt_ref = []
    num_ped_list = []
    min_mn_idx = []

    pbar = tqdm(total=len(test_loader.dataset))
    pbar.set_description('Testing epoch %d'%(epoch+1))
    for batch_idx, batch_data in enumerate(test_loader):    
        model.test_seen = model.test_seen + len(batch_data['num_ped'])

        goal_pred_shift, traj_pred_shift, traj_gt_shift, traj_xyt_ref, goal_fde, min_ade, min_fde, min_goal_fde_mn_idx = model.inference_hybrid(batch_data, min_goal_fde_mn_idx)        
        # print(min_goal_fde_mn_idx.size())
        # tmp = input()
        all_goal_pred_shift.append(goal_pred_shift.detach().cpu())
        all_traj_pred_shift.append(traj_pred_shift.detach().cpu())
        all_traj_gt_shift.append(traj_gt_shift.detach().cpu())
        all_traj_gt_ref.append(traj_xyt_ref.detach().cpu())
        min_mn_idx.append(min_goal_fde_mn_idx.detach().cpu())
        num_ped_list.extend(batch_data['num_ped'])

        pbar.update(len(batch_data['num_ped']))

    pbar.close()

    all_goal_pred_shift = torch.cat(all_goal_pred_shift, dim=2)
    all_traj_pred_shift = torch.cat(all_traj_pred_shift, dim=2)
    all_traj_gt_shift = torch.cat(all_traj_gt_shift, dim=2)
    all_traj_gt_ref = torch.cat(all_traj_gt_ref, dim=2)
    min_mn_idx = torch.cat(min_mn_idx, dim=0)
    
    return (all_goal_pred_shift, all_traj_pred_shift, all_traj_gt_shift, all_traj_gt_ref, num_ped_list, min_mn_idx)       