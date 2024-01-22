from utils.Frangi_filter import Frangi
from utils.Crop_box import Crop_pre
from utils.Get_patch import Get_patch
from utils.utils import get_csv_split
import yaml
import multiprocessing
import argparse
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 预处理
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--Direct_model',type=str,default='FCN')
    p.add_argument('--Direct_parameter',type=str,default='Mid_resolution_4_Dice')
    p.add_argument('--pools',type=int,default=4)
    p.add_argument('--patch_size',type=int,default=32)

    args = p.parse_args()
    k = args.fold
    config_file = args.config_file
    coarse_version=args.Direct_model
    direct_parameters =args.Direct_parameter
    pool_num=args.pools
    patch_size=args.patch_size

    # 根据预分割进行裁剪
    with open(config_file) as f:
        config = yaml.load(f)

    img_path = config['General_parameters']['data_path']
    csv_path = config['General_parameters']['csv_path']
    mid_path = config['General_parameters']['mid_path']

    crop_path = os.path.join(mid_path,'Patches',coarse_version,direct_parameters,'fold_%d'% k,'crop')
    patch_path = os.path.join(mid_path,'Patches',coarse_version,direct_parameters,'fold_%d'% k)
    p_path = os.path.join('result/Direct_seg',coarse_version,direct_parameters,'fold_%d' % k, 'pre_label')
    save_enhance=os.path.join(mid_path,'Frangi')

    id_dict = get_csv_split(csv_path, k)
    id_list = id_dict['train'] + id_dict['valid']

    if not os.path.exists(save_enhance):
        ## 滤波
        print('Frangi..........')
        frangi_opt = Frangi(img_path, save_enhance)
        p = multiprocessing.Pool(pool_num)
        p.map_async(frangi_opt.run_enhance, id_list)
        p.close()
        p.join()

    # # 裁剪
    if not os.path.exists(os.path.join(mid_path,'patches',coarse_version,direct_parameters,'fold_%d'% k,'crop_fold_%d.npy'%k)):
        print('crop...........')
        crop_opt = Crop_pre(p_path, img_path, crop_path, save_enhance)
        p = multiprocessing.Pool(pool_num)
        crop_box = p.map(crop_opt.run_crop, id_list)
        p.close()
        p.join()

        record_box = {}
        record_box['ID'] = id_list
        record_box['box'] = crop_box
        np.save(os.path.join(mid_path,'patches',coarse_version,direct_parameters,'fold_%d'% k,'crop_fold_%d.npy'%k), record_box)

    # 获取体素块
    for p_size in [patch_size]:
        for dt in ['train', 'valid']:
            print('get_patch %s %d' % (dt, p_size))
            get_patch_opt = Get_patch(crop_path, crop_path, crop_path, patch_path, p_size, dt)
            p = multiprocessing.Pool(pool_num)
            p.map(get_patch_opt.run_main, id_dict[dt])
            p.close()
            p.join()
