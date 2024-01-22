import os
from model.GCN import GraphSAGE
from data.Graph_loader import Graph_loader, collate, Inference_graph
from utils.Make_graph import recover_node, img_resample, Recover_label
from tqdm import tqdm
import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from model.loss import Dist_loss
import argparse
from dgl.data.utils import save_graphs, load_graphs
from utils.utils import get_csv_split
import nibabel as nib
import numpy as np
import re
import multiprocessing
from utils.Calculate_metrics import Cal_metrics
import pandas as pd
import time


def train(model, criterion, train_loader, opt, device, e):
    model.train()
    train_sum = 0
    with tqdm(train_loader) as t:
        for index, batch in enumerate(t):
            g = batch.to(device)
            xv = g.ndata['xv'][:, :32]
            xv = xv.float()
            rv = g.ndata['rv']
            xv = xv.to(device)
            rv = rv.to(device)
            outputs = model(g, xv)

            opt.zero_grad()

            loss = criterion(outputs, rv)
            train_sum += loss.item()
            loss.backward()

            opt.step()

            t.set_description("Epoch %i" % e)
            # t.set_postfix(train_loss=(train_sum/(index+1)))
            t.set_postfix(train_loss=train_sum / (index + 1))

    return train_sum / len(train_loader)


def valid(model, criterion, valid_loader, opt, device, e):
    model.eval()
    valid_sum = 0
    with tqdm(valid_loader) as t:
        for index, batch in enumerate(t):
            g = batch.to(device)
            xv = g.ndata['xv'][:, :32]
            xv = xv.float()
            rv = g.ndata['rv']
            xv = xv.to(device)
            rv = rv.to(device)

            with torch.no_grad():
                outputs = model(g, xv)
                loss = criterion(outputs, rv)
            valid_sum += loss.item()

            t.set_description("Epoch %i" % e)
            t.set_postfix(valid_loss=valid_sum / (index + 1))

    return valid_sum / len(valid_loader)


def inference(model, criterion, train_loader, valid_loader, opt, device, save_img_path,str_key='pre_rv'):
    print('=============================inference==============================')
    model.eval()
    train_sum = 0
    valid_sum = 0
    is_save_train = False
    if is_save_train:
        with tqdm(train_loader) as t:
            for index, batch in enumerate(t):
                id_index = batch['id_index']
                g = batch['g'].to(device)
                g = g.to(device)
                xv = g.ndata['xv'][:, :32]
                xv = xv.float()
                rv = g.ndata['rv']
                xv = xv.to(device)
                rv = rv.to(device)

                with torch.no_grad():
                    outputs = model(g, xv)
                    loss = criterion(outputs, rv)
                train_sum += loss.item()

                t.set_postfix(valid_loss=train_sum / (index + 1))

                g.ndata['pre_rv'] = outputs

                save_graphs(os.path.join(save_img_path, id_index), [g])

    with tqdm(valid_loader) as t:
        for index, batch in enumerate(t):
            id_index = batch['id_index']
            g = batch['g']
            g = g.to(device)
            xv = g.ndata['xv'][:, :32]
            xv = xv.float()
            rv = g.ndata['rv']
            xv = xv.to(device)
            rv = rv.to(device)

            with torch.no_grad():
                outputs = model(g, xv)
                loss = criterion(outputs, rv)
            valid_sum += loss.item()

            t.set_postfix(valid_loss=valid_sum / (index + 1))

            g.ndata[str_key] = outputs

            save_graphs(os.path.join(save_img_path, id_index), [g])


def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--gpu_index', type=int, default=7)
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--load_num', type=int, default=30)
    p.add_argument('--is_train', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--Direct_model', type=str, default='FCN')
    p.add_argument('--model', type=str, default='GCN')
    p.add_argument('--pools', type=int, default=32)
    p.add_argument('--is_inference', type=int, default=1)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--Direct_parameter', type=str, default='Mid_resolution_4_Dice')
    return p.parse_args()


if __name__ == '__main__':
    args = args_input()
    config_file = args.config_file
    k = args.fold
    load_num = args.load_num
    batch_size = args.batch_size
    version = args.model
    p_size = args.patch_size
    is_infer = args.is_inference
    coarse_version = args.Direct_model
    direct_parameters = args.Direct_parameter
    pool_num = args.pools
    epochs=args.epochs

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_save_path = r'result/Graph_seg/%s_%s/fold_%d/model_save' % (coarse_version, version, k)
    # save_pre_path = r'result/Graph_seg/%s_%s/fold_%d/graph_save' % (coarse_version, version, k)
    # save_label_path = r'result/Graph_seg/%s_%s/fold_%d/pre_label' % (coarse_version, version, k)

    save_graph_path = 'result/Graph_seg/%s/%s/%s/fold_%d/graph/save_graph' % (
    coarse_version, direct_parameters, version, k)
    save_model_path = 'result/Graph_seg/%s/%s/%s/fold_%d/graph/model_save' % (
    coarse_version, direct_parameters, version, k)
    recover_path = 'result/Graph_seg/%s/%s/%s/fold_%d/graph/pre_label' % (coarse_version, direct_parameters, version, k)

    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(save_graph_path, exist_ok=True)
    os.makedirs(recover_path, exist_ok=True)

    print('model:%s || fold_%d' % (version, k))
    print('load_num:%d' % load_num)

    # 读取参数配置文件
    with open(config_file) as f:
        config = yaml.load(f)
        data_path = config['General_parameters']['data_path']
        # epochs = config['General_parameters']['epoch']
        csv_path = config['General_parameters']['csv_path']
        mid_path = config['General_parameters']['mid_path']
        learning_rate= config['General_parameters']['lr']

    train_path = os.path.join(mid_path, 'Graph', coarse_version, direct_parameters, 'fold_%d' % k, 'train')
    valid_path = os.path.join(mid_path, 'Graph', coarse_version, direct_parameters, 'fold_%d' % k, 'valid')

    spacing = np.array([0.5, 0.5, 0.5]).reshape((1, 3))

    # 数据加载
    train_set = Graph_loader(train_path)
    valid_set = Graph_loader(valid_path)
    train_loader = DataLoader(train_set, batch_size, collate_fn=collate, num_workers=32)
    valid_loader = DataLoader(valid_set, batch_size, collate_fn=collate, num_workers=32)

    # network
    net = GraphSAGE(32, 64, 1, 3, None, 0.5, 'gcn').to(device)

    if re.search(pattern=r'GCN', string=version):
        print('GCN')
        net = GraphSAGE(32, 64, 1, 3, None, 0.5, 'gcn').to(device)
    else:
        net = GraphSAGE(32, 64, 1, 3, None, 0.5, 'gcn').to(device)

    # load_num model

    if load_num != 0:
        net.load_state_dict(torch.load(save_model_path + '/net_%d.pkl' % load_num, map_location='cuda:%d' % 0))
        load_num = load_num + 1

    # 优化器
    net_opt = Adam(net.parameters(), lr=learning_rate)

    # 损失函数
    criterion = Dist_loss()

    train_loss_set = []
    valid_loss_set = []
    epoch_list = []

    if args.is_train == 1:
        for e in range(load_num, epochs):
            train_loss = train(net, criterion, train_loader, net_opt, device, e)
            valid_loss = valid(net, criterion, valid_loader, net_opt, device, e)
            train_loss_set.append(train_loss)
            valid_loss_set.append(valid_loss)
            epoch_list.append(e)
            if e % 3 == 0:
                torch.save(net.state_dict(), save_model_path + '/net_%d.pkl' % e)

        record = dict()
        record['epoch'] = epoch_list
        record['train_loss'] = train_loss_set
        record['valid_loss']=valid_loss_set
        record = pd.DataFrame(record)
        record_name = time.strftime("%Y_%m_%d_%H.csv", time.localtime())
        record.to_csv(r'result/Graph_seg/%s/%s/%s/fold_%d/graph/%s' % (
        coarse_version, direct_parameters, version, k, record_name), index=False)

    # 推断
    infer_train_set = Inference_graph(train_path)
    infer_valid_set = Inference_graph(valid_path)

    # net.load_state_dict(torch.load(model_save_path + '/net_69.pkl'))
    if is_infer:
        inference(net, criterion, infer_train_set, infer_valid_set, net_opt, device, save_graph_path)

    id_dict = get_csv_split(csv_path, k)

    # Recover
    print('Recover .........')
    RL = Recover_label(data_path, save_graph_path, recover_path, spacing, 'pre_rv')
    p = multiprocessing.Pool(pool_num)
    p.map(RL.run, id_dict['valid'])
    p.close()
    p.join()

    # Calculate_dice
    print('Calculate.........')
    CD = Cal_metrics(recover_path, data_path, pre_label_name='pre_rv.nii.gz')
    p = multiprocessing.Pool(pool_num)
    result = p.map(CD.calculate_dice, id_dict['valid'])
    p.close()
    p.join()

    record_dice = dict()
    record_dice['ID'] = id_dict['valid']
    result = np.array(result)
    record_dice['dice'] = result[:, 0]
    record_dice['ahd'] = result[:, 1]
    record_dice['hd'] = result[:, 2]
    record_dice = pd.DataFrame(record_dice)
    record_dice.to_csv(
        r'result/Graph_seg/%s/%s/%s/fold_%d/graph/result.csv' % (coarse_version, direct_parameters, version, k),
        index=False)
