import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from os.path import join
from os import listdir
import multiprocessing
from utils.utils import get_csv_split
import argparse
import yaml
import os
from utils.Calculate_metrics import Cal_metrics
import pandas as pd
import re


class Ensemble_concate:
    def __init__(self, coarse_path, patch_path, patch_list, save_path, is_add_coarse, save_name):
        self.patch_path = patch_path
        self.coarse_path = coarse_path
        self.patch_list = patch_list
        self.save_path = save_path
        self.is_add = is_add_coarse
        self.save_name = save_name

    def process(self, id):
        print(id)
        c_path = os.path.join(self.coarse_path, 'pre_label', id, 'pre_label.nii.gz')
        c_nii = nib.load(c_path)
        coarse = c_nii.get_fdata()
        ensemble = np.zeros_like(coarse)
        if self.is_add == 1:
            ensemble = ensemble + coarse
            flag = 1
        else:
            flag = 0
        for i in self.patch_list:
            p_path = os.path.join(self.patch_path, 'patch_%d' % i, 'pre_label', id, 'pre_label.nii.gz')
            # print('patch')
            patch = nib.load(p_path).get_fdata()
            patch[patch >= 0.5] = 1
            patch[patch < 0.5] = 0
            ensemble = ensemble + patch

        ensemble[ensemble < ((len(self.patch_list) + flag) // 2)] = 0
        ensemble[ensemble >= ((len(self.patch_list) + flag) // 2)] = 1

        os.makedirs(os.path.join(self.save_path, id), exist_ok=True)
        nib.save(nib.Nifti1Image(ensemble.astype(np.float), c_nii.affine),
                 os.path.join(self.save_path, id, '%s' % self.save_name))
        print('%s:done' % id)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--Direct_model', type=str, default='FCN')
    p.add_argument('--Direct_parameter', type=str, default='Mid_resolution_4_Dice')
    p.add_argument('--global_seg', type=str, default='Mid_resolution_4_Dice')
    p.add_argument('--pools', type=int, default=16)
    p.add_argument('--add_global', type=int, default=1)

    args = p.parse_args()
    k = args.fold
    config_file = args.config_file
    coarse_version = args.Direct_model
    direct_parameters = args.Direct_parameter
    global_seg = args.global_seg
    pool_num = args.pools
    is_add = args.add_global
    parameter_record = direct_parameters

    print('coarse_version:', coarse_version)

    # 根据预分割进行裁剪
    with open(config_file) as f:
        config = yaml.load(f)

    img_path = config['General_parameters']['data_path']
    csv_path = config['General_parameters']['csv_path']
    mid_path = config['General_parameters']['mid_path']

    global_seg_path = os.path.join('result/Direct_seg', coarse_version, global_seg, 'fold_%d' % k)
    patch_seg_path = r'result/Prior_Patch_seg/%s/%s/fold_%d' % (coarse_version, parameter_record, k)
    save_path = r'result/Prior_Patch_seg/%s/%s/fold_%d/ensemble' % (coarse_version, parameter_record, k)

    ID_dict = get_csv_split(csv_path, k)
    ID_list = ID_dict['valid']

    Ec = Ensemble_concate(global_seg_path, patch_seg_path, [16, 32, 64], save_path, is_add, 'add_%d.nii.gz' % is_add)

    # Ensemble
    print('Ensemble............')
    p = multiprocessing.Pool(pool_num)
    p.map(Ec.process, ID_list)
    p.close()
    p.join()

    # Calculate_dice
    print('Calculate_dice.......')
    Cd = Cal_metrics(save_path, img_path, pre_label_name='add_%d.nii.gz' % is_add)
    p = multiprocessing.Pool(pool_num)
    res = p.map(Cd.calculate_dice, ID_list)
    p.close()
    p.join()

    dice_list = []
    ahd_list = []
    hd_list = []

    for dice, ahd, hd in res:
        dice_list.append(dice)
        ahd_list.append(ahd)
        hd_list.append(hd)

    record_dice = {}
    record_dice['ID'] = ID_list
    record_dice['dice'] = dice_list
    record_dice['ahd'] = ahd_list
    record_dice['hd'] = hd_list
    record_dice = pd.DataFrame(record_dice)
    record_dice.to_csv(
        r'result/Prior_Patch_seg/%s/%s/fold_%d/ensemble/result_%d.csv' % (coarse_version, parameter_record, k, is_add),
        index=False)
