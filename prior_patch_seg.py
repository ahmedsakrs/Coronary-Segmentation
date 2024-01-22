import torch
import argparse
import os
import numpy as np
import nibabel as nib
import torch.optim as optim
import torch.nn as nn
import multiprocessing
import pandas as pd
import yaml
import time
from model.CNN_model import Unet, Unet_Patch, NestedUNet3d
from tqdm import tqdm
from utils.parallel import parallel
from utils.Calculate_metrics import Cal_metrics
from utils.Recover_patch import Recover_patch
from utils.utils import Transform
from data.Patch_loader import CoronaryImagePatch, CoronaryImageEnhance
from utils.utils import get_csv_split
from torch.utils.data import DataLoader
from model.loss import DiceLoss
from utils.Crop_box import Recover_Crop


def train(model, criterion, train_loader, opt, device, e):
    model.train()
    train_sum = 0
    for j, batch in enumerate(train_loader):
        img, label = batch['img'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)
        outputs = model(img)
        opt.zero_grad()
        loss = criterion(outputs, label)
        print('Epoch {:<3d}  |  Step {:>3d}/{:<3d}  | train loss {:.4f}'.format(e, j, len(train_loader), loss.item()))
        train_sum += loss.item()
        loss.backward()
        opt.step()
    print('average_train_loss: %f' % (train_sum / len(train_loader)))
    return train_sum / len(train_loader)


def valid(model, criterion, valid_loader, device, e):
    model.eval()
    valid_sum = 0
    for j, batch in enumerate(valid_loader):
        img, label = batch['img'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
            loss = criterion(outputs, label)
        valid_sum += loss.item()
        print('Epoch {:<3d}  |Step {:>3d}/{:<3d}  | valid loss {:.4f}'.format(e, j, len(valid_loader), loss.item()))
    print('average_valid_loss: %f' % (valid_sum / len(valid_loader)))
    return valid_sum / len(valid_loader)


def inference(model, train_loader, valid_loader, device, save_img_path, is_infer_train=False):
    # 得到预测的标签以及重构
    model.eval()
    if is_infer_train:
        for batch in tqdm(train_loader):
            img, label = batch['img'].float(), batch['label'].float()
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                outputs = model(img)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(1)
            pre = outputs.cpu().detach().numpy()
            ID = batch['image_index']
            affine = batch['affine']
            os.makedirs(os.path.join(save_img_path), exist_ok=True)
            batch_save(ID, affine, pre, save_img_path)

    for batch in tqdm(valid_loader):
        img, label = batch['img'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)
        # file_name = batch['id_index'][0]

        with torch.no_grad():
            outputs = model(img)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze(1)
        pre = outputs.cpu().detach().numpy()
        ID = batch['image_index']
        affine = batch['affine']
        os.makedirs(os.path.join(save_img_path), exist_ok=True)
        batch_save(ID, affine, pre, save_img_path)


def batch_save(ID, affine, pre, save_img_path):
    batch_size = len(ID)
    save_list = [save_img_path] * batch_size
    parallel(save_picture, pre, affine, save_list, ID, thread=True)


def save_picture(pre, affine, save_name, id):
    pre = pre
    pre_label = pre
    nib.save(nib.Nifti1Image(pre_label, affine), os.path.join(save_name, id + '.nii.gz'))


def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--gpu_index', type=int, default=0)
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--load_num', type=int, default=30)
    p.add_argument('--is_train', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--Direct_model', type=str, default='FCN')
    p.add_argument('--channel', type=int, default=4)
    p.add_argument('--pools', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--is_inference', type=int, default=1)
    p.add_argument('--loss', type=str, default='Dice')
    p.add_argument('--flip_prob', type=float, default=0)
    p.add_argument('--rotate_prob', type=float, default=0)
    # p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--Direct_parameter', type=str, default='Mid_resolution_4_Dice')
    p.add_argument('--epochs', type=int, default=30)
    return p.parse_args()


if __name__ == '__main__':
    args = args_input()
    config_file = args.config_file
    k = args.fold
    load_num = args.load_num
    batch_size = args.batch_size
    p_size = args.patch_size
    coarse_version = args.Direct_model
    flip_prob = args.flip_prob
    rotate_prob = args.rotate_prob
    direct_parameters = args.Direct_parameter
    epochs=args.epochs
    parameter_record = '%s' % direct_parameters

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_save_path = r'result/Prior_Patch_seg/%s/%s/fold_%d/patch_%d/model_save' % (
        coarse_version, parameter_record, k, p_size)
    pre_patch_path = r'result/Prior_Patch_seg/%s/%s/fold_%d/patch_%d/pre_patch' % (
        coarse_version, parameter_record, k, p_size)
    pre_label_path = r'result/Prior_Patch_seg/%s/%s/fold_%d/patch_%d/pre_label' % (
        coarse_version, parameter_record, k, p_size)

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(pre_patch_path, exist_ok=True)
    os.makedirs(pre_label_path, exist_ok=True)

    print('coarse: %s_%s|| model:%s || fold_%d' % (coarse_version, direct_parameters, parameter_record, k))
    print('load_num:%d' % load_num)

    # 从config.yaml里面读取参数
    with open(r'config/config.yaml') as f:
        config = yaml.load(f)
        # patch_type=config['General_parameters'][overlap]
        img_path = config['General_parameters']['data_path']
        csv_path = config['General_parameters']['csv_path']
        mid_path = config['General_parameters']['mid_path']

    patch_path = os.path.join(mid_path, 'Prior_Patches', coarse_version, direct_parameters, 'fold_%d' % k)

    train_data_path = os.path.join(patch_path, 'patch_%d' % p_size, 'train_img_patch')
    train_label_path = os.path.join(patch_path, 'patch_%d' % p_size, 'train_label_patch')
    train_enhance_path = os.path.join(patch_path, 'patch_%d' % p_size, 'train_enhance_patch')

    valid_data_path = os.path.join(patch_path, 'patch_%d' % p_size, 'valid_img_patch')
    valid_label_path = os.path.join(patch_path, 'patch_%d' % p_size, 'valid_label_patch')
    valid_enhance_path = os.path.join(patch_path, 'patch_%d' % p_size, 'valid_enhance_patch')

    csv_record_path = os.path.join(patch_path, 'patch_%d' % p_size, 'csv_patch_record')

    ID_list = get_csv_split(csv_path, k)

    net = NestedUNet3d(1, 1).to(device)
    Data_set = CoronaryImagePatch
    trans = Transform(flip_prob, rotate_prob)

    # 数据加载
    train_set = Data_set(train_data_path, train_label_path, train_enhance_path, transform=trans.transform)
    valid_set = Data_set(valid_data_path, valid_label_path, valid_enhance_path, transform=trans.transform)
    train_loader = DataLoader(train_set, batch_size, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size, num_workers=8, shuffle=False)

    model_list = os.listdir(model_save_path)
    if load_num == 0:
        for m in net.modules():
            if isinstance(m, (nn.Conv3d)):
                nn.init.orthogonal(m.weight)
    else:
        net.load_state_dict(torch.load(model_save_path + '/net_%d.pkl' % load_num, map_location='cuda:%d' % 0))
        load_num = load_num + 1

    net_opt = optim.Adam(net.parameters(), lr=0.001)
    criterion = DiceLoss()

    train_loss_set = []
    valid_loss_set = []
    epoch_list = []

    ## 训练
    if args.is_train == 1:
        for e in range(load_num, epochs):
            t0 = time.time()
            train_loss = train(net, criterion, train_loader, net_opt, device, e)
            # valid_loss = valid(net, criterion, valid_loader, device, e)
            train_loss_set.append(train_loss)
            # valid_loss_set.append(valid_loss)
            epoch_list.append(e)
            t1 = time.time()
            # print("train_loss:%f || valid_loss:%f" % (train_loss, valid_loss))
            print("train_loss:%f || times:%f" % (train_loss, t1 - t0))
            if e % 5 == 0:
                torch.save(net.state_dict(), model_save_path + '/net_%d.pkl' % e)
        record = dict()
        record['epoch'] = epoch_list
        record['train_loss'] = train_loss_set
        # record['valid_loss']=valid_loss_set
        record = pd.DataFrame(record)
        record_name = time.strftime("%Y_%m_%d_%H.csv", time.localtime())
        record.to_csv(r'result/Prior_Patch_seg/%s/%s/fold_%d/patch_%d/%s' % (
            coarse_version, parameter_record, k, p_size, record_name), index=False)

    # 推断
    print("inference.....")
    train_set = Data_set(train_data_path, train_label_path,train_enhance_path , transform=None)
    valid_set = Data_set(valid_data_path, valid_label_path,valid_enhance_path,  transform=None)
    train_infer_loader = DataLoader(train_set, batch_size * 2, shuffle=False, num_workers=8)
    valid_infer_loader = DataLoader(valid_set, batch_size, shuffle=False, num_workers=32)

    inference(net, train_infer_loader, valid_infer_loader, device, pre_patch_path)

    print('Recover Patch......')
    recover = Recover_patch(p_size, pre_patch_path, csv_record_path,
                            pre_label_path, img_path, save_file_name='pre_label.nii.gz')
    p = multiprocessing.Pool(48)
    p.map(recover.run_recover, ID_list['valid'])
    p.close()
    p.join()

    print('calculate dice.......')
    # crop_path = os.path.join(crop_path, coarse_version, 'crop_fold_%d.npy' % k)
    CD = Cal_metrics(pre_label_path, img_path, p_size)
    p = multiprocessing.Pool(48)
    result = p.map(CD.calculate_dice, ID_list['valid'])
    p.close()
    p.join()

    # 保存结果
    record_dice = dict()
    record_dice['ID'] = ID_list['valid']
    result = np.array(result)
    record_dice['dice'] = result[:, 0]
    record_dice['ahd'] = result[:, 1]
    record_dice['hd'] = result[:, 2]
    record_dice = pd.DataFrame(record_dice)
    record_dice.to_csv(r'result/Prior_Patch_seg/%s/%s/fold_%d/patch_%d/result.csv' % (
        coarse_version,  parameter_record, k, p_size), index=False)
