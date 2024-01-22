from data.Tree_loader import Tree_Batch, valid_batcher, train_batcher, Tree_inference
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import argparse
from dgl.data.utils import save_graphs
from utils.utils import get_csv_split
from model.loss import DiceLoss
import re
from model.TreeConvRNN import TreeConvLSTM3d, TreeConvGRU3d
import yaml
import pandas as pd
import time
from utils.Make_tree import Recover_img
from utils.Calculate_metrics import Cal_metrics
import multiprocessing
import numpy as np


def train(model, criterion, train_loader, opt, device, e, version, patch_size):
    model.train()
    train_sum = 0
    h1 = int(patch_size[0]) // 4
    h2 = int(patch_size[1]) // 4
    h3 = int(patch_size[2]) // 4
    count = 0
    with tqdm(train_loader) as t:
        for batch in t:
            count += 1
            g = batch.graph
            n = g.number_of_nodes()
            h = torch.zeros((n, 10, h1, h2, h3)).to(device)
            g.ndata['data'] = g.ndata['data'].to(device)
            g.ndata['label'] = g.ndata['label'].to(device)
            if version == "TreeConvLSTM":
                outputs = model(g, h, h)
            else:
                outputs = model(g, h)
            opt.zero_grad()
            loss = criterion(outputs, batch.label)
            t.set_description("%s_%d_Epoch %i" % (version, k, e))
            t.set_postfix(tloss=train_sum / count)
            train_sum += loss.item()
            loss.backward()
            opt.step()
    return train_sum / len(train_loader)


def valid(model, criterion, loader, device, e, version, patch_size):
    model.eval()
    valid_sum = 0
    h1 = int(patch_size[0]) // 4
    h2 = int(patch_size[1]) // 4
    h3 = int(patch_size[2]) // 4
    count = 0

    with tqdm(loader) as t:
        for batch in t:
            # img,label=batch['img'].float(),batch['label'].float()
            # img,label=img.to(device),label.to(device)
            count += 1

            g = batch.graph
            n = g.number_of_nodes()
            h = torch.zeros((n, 10, h1, h2, h3)).to(device)
            g.ndata['data'] = g.ndata['data'].to(device)
            g.ndata['label'] = g.ndata['label'].to(device)
            with torch.no_grad():
                if version == "TreeConvLSTM":
                    outputs = model(g, h, h)
                else:
                    outputs = model(g, h)
            loss = criterion(outputs, batch.label)
            t.set_description("%s_%d_Epoch %i" % (version, k, e))

            # g.ndata['pre_label']=torch.sigmoid(outputs.detach())
            # print('Epoch {:<3d}  |  Step {:>3d}/{:<3d}  | train loss {:.4f}'.format(e,j, len(train_loader), loss.item()))
            valid_sum += loss.item()
            t.set_postfix(valid_loss=valid_sum / count)

    return valid_sum / len(train_loader)


def inference(model, criterion, train_loader, valid_loader, device, save_img_path, patch_size, version, is_train=False):
    model.eval()
    valid_sum = 0
    train_sum = 0
    h1 = patch_size[0] // 4
    h2 = patch_size[1] // 4
    h3 = patch_size[2] // 4
    if is_train:
        with tqdm(train_loader) as t:
            for batch in t:
                g = batch['g'].to(device)
                n = g.number_of_nodes()
                h = torch.zeros((n, 10, h1, h2, h3)).to(device)
                g.ndata['data'] = g.ndata['data'].to(device)
                g.ndata['label'] = g.ndata['label'].to(device)
                with torch.no_grad():
                    if re.search('LSTM', version):
                        outputs = model(g, h, h)
                    else:
                        outputs = model(g, h)
                loss = criterion(outputs, g.ndata['label'])
                t.set_postfix(train_loss=loss.item())
                g.ndata['pre_label'] = torch.sigmoid(outputs.detach())
                valid_sum += loss.item()
                file_name = batch['id_index']
                save_graphs(os.path.join(save_img_path, file_name), [g])

    with tqdm(valid_loader) as t:
        for batch in t:
            g = batch['g'].to(device)
            n = g.number_of_nodes()
            h = torch.zeros((n, 10, h1, h2, h3)).to(device)
            g.ndata['data'] = g.ndata['data'].to(device)
            g.ndata['label'] = g.ndata['label'].to(device)
            with torch.no_grad():
                if re.search('LSTM', version):
                    outputs = model(g, h, h)
                else:
                    outputs = model(g, h)
            loss = criterion(outputs, g.ndata['label'])
            t.set_postfix(valid_loss=loss.item())
            g.ndata['pre_label'] = torch.sigmoid(outputs.detach())
            valid_sum += loss.item()
            file_name = batch['id_index']
            save_graphs(os.path.join(save_img_path, file_name), [g])


def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--gpu_index', type=int, default=0)
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--load_num', type=int, default=0)
    p.add_argument('--is_train', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--z_size', type=int, default=4)
    p.add_argument('--Direct_model', type=str, default='FCN')
    p.add_argument('--model', type=str, default='TreeConvLSTM')
    p.add_argument('--pools', type=int, default=32)
    p.add_argument('--is_inference', type=int, default=1)
    p.add_argument('--loss', type=str, default='Dice')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--Direct_parameter', type=str, default='Mid_resolution_4_Dice')
    return p.parse_args()


if __name__ == '__main__':
    # 设置命令行参数
    args = args_input()
    config_file = args.config_file
    k = args.fold
    load_num = args.load_num
    b_size = args.batch_size
    version = args.model
    p_size = args.patch_size
    pool_num = args.pools
    gpu_index = args.gpu_index
    mode = args.is_train
    coarse_version = args.Direct_model
    direct_parameters = args.Direct_parameter
    epochs=args.epochs
    patch_size = [args.patch_size, args.patch_size, args.z_size]

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda:%d"%gpu_index if torch.cuda.is_available() else "cpu")

    with open(r'config/config.yaml') as f:
        config = yaml.load(f)
        csv_path = config['General_parameters']['csv_path']
        data_path = config['General_parameters']['data_path']
        mid_path = config['General_parameters']['mid_path']
        # patch_path=config['General_parameters']['patch_path']

    id_dict = get_csv_split(csv_path, k)

    # 设置结果路径
    save_graph_path = 'result/Tree_seg/%s/%s/%s/fold_%d/patch_%d_%d_%d/save_graph' % (
    coarse_version, direct_parameters, version, k, patch_size[0], patch_size[1], patch_size[2])
    save_model_path = 'result/Tree_seg/%s/%s/%s/fold_%d/patch_%d_%d_%d/model_save' % (
    coarse_version, direct_parameters, version, k, patch_size[0], patch_size[1], patch_size[2])
    recover_path = 'result/Tree_seg/%s/%s/%s/fold_%d/patch_%d_%d_%d/pre_label' % (
    coarse_version, direct_parameters, version, k, patch_size[0], patch_size[1], patch_size[2])

    os.makedirs(save_graph_path, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(recover_path, exist_ok=True)

    # 加载数据
    train_path = os.path.join(mid_path, 'Tree', coarse_version, direct_parameters, 'fold_%d' % k,
                              'patch_%d_%d_%d' % (patch_size[0], patch_size[1], patch_size[2]), 'train')
    valid_path = os.path.join(mid_path, 'Tree', coarse_version, direct_parameters, 'fold_%d' % k,
                              'patch_%d_%d_%d' % (patch_size[0], patch_size[1], patch_size[2]), 'valid')

    train_data = Tree_Batch(train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=b_size, shuffle=True, collate_fn=train_batcher(device),
                              num_workers=0)

    valid_data = Tree_Batch(valid_path)
    valid_loader = DataLoader(dataset=valid_data, batch_size=b_size, shuffle=True, collate_fn=valid_batcher(device),
                              num_workers=0)

    if version == "TreeConvLSTM":
        net = TreeConvLSTM3d(20, 10, patch_size).to(device)
    elif version == "TreeConvGRU":
        net = TreeConvGRU3d(20, 10).to(device)
    else:
        raise ValueError('no model')

    if load_num != 0:
        net.load_state_dict(torch.load(save_model_path + '/net_%d.pkl' % load_num))

    net_opt = Adam(net.parameters(), lr=0.001)
    criterion = DiceLoss()

    # 训练

    train_loss_set = []
    valid_loss_set = []
    epoch_list = []

    if mode == 1:
        for e in range(load_num, epochs):
            train_loss = train(net, criterion, train_loader, net_opt, device, e, version, patch_size)
            valid_loss = valid(net, criterion, valid_loader, device, e, version, patch_size)
            epoch_list.append(e)
            train_loss_set.append(train_loss)
            if e % 3 == 0:
                torch.save(net.state_dict(), save_model_path + '/net_%d.pkl' % e)
        record = dict()
        record['epoch'] = epoch_list
        record['train_loss'] = train_loss_set
        # record['valid_loss'] = valid_loss_set

        record = pd.DataFrame(record)
        record_name = time.strftime("%Y_%m_%d_%H.csv", time.localtime())
        record.to_csv('result/Tree_seg/%s/%s/%s/fold_%d/patch_%d_%d_%d/%s' % (
        coarse_version, direct_parameters, version, k, patch_size[0], patch_size[1], patch_size[2], record_name),
                      index=False)

    # inference
    print('inference .....')
    train_infer = Tree_inference(train_path)
    valid_infer = Tree_inference(valid_path)
    inference(net, criterion, train_infer, valid_infer, device, save_graph_path, patch_size, version)

    # 复原图像
    print('recover .......')
    recover_opt = Recover_img(data_path, recover_path, save_graph_path,save_file_name='pre_label.nii.gz')
    p = multiprocessing.Pool(pool_num)
    p.map(recover_opt.recover_img_run, id_dict['valid'])
    p.close()
    p.join()

    print('calculate dice.......')
    CD = Cal_metrics(recover_path, data_path, p_size)
    p = multiprocessing.Pool(pool_num)
    result = p.map(CD.calculate_dice, id_dict['valid'])
    p.close()
    p.join()

    record_dice=dict()
    record_dice['ID'] = id_dict['valid']
    result=np.array(result)
    record_dice['dice'] = result[:,0]
    record_dice['ahd']=result[:,1]
    record_dice['hd']=result[:,2]
    record_dice = pd.DataFrame(record_dice)
    record_dice.to_csv(r'result/Tree_seg/%s/%s/%s/fold_%d/patch_%d_%d_%d/result.csv' %
                       (coarse_version, direct_parameters, version, k, patch_size[0], patch_size[1], patch_size[2]), index=False)
