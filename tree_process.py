from utils.Make_tree import Convert_tree
from utils.utils import get_csv_split
import yaml
import os
import nibabel as nib
import numpy as np
import multiprocessing
import argparse
from utils.Calculate_metrics import get_region_num
import time

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--Direct_model', type=str, default='FCN')
    p.add_argument('--Direct_parameter', type=str, default='Mid_resolution_4_Dice')
    p.add_argument('--pools', type=int, default=4)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--z_size', type=int, default=4)

    args = p.parse_args()
    k = args.fold
    config_file = args.config_file
    coarse_version = args.Direct_model
    direct_parameters = args.Direct_parameter
    pool_num = args.pools
    patch_size = args.patch_size
    z_size = args.z_size

    print('coarse_version: %s || fold: %d' % (coarse_version, k))

    with open(r'config/config.yaml') as f:
        config = yaml.load(f)

    img_path = config['General_parameters']['data_path']
    csv_path = config['General_parameters']['csv_path']
    mid_path = config['General_parameters']['mid_path']

    tree_path = os.path.join(mid_path, 'Tree', coarse_version, direct_parameters, 'fold_%d' % k)
    pre_seg_path = os.path.join('result/Direct_seg', coarse_version, direct_parameters, 'fold_%d' % k, 'pre_label')

    id_dict = get_csv_split(csv_path, k)
    id_list = id_dict['train'] + id_dict['valid']

    # 3d块转换
    t1 = time.time()

    for dt in ['train', 'valid']:
        convert_tree = Convert_tree(pre_seg_path, img_path, img_path, tree_path, dt, (patch_size, patch_size, z_size))
        print('convert_tree %s' % dt)
        p = multiprocessing.Pool(pool_num)
        p.map(convert_tree.run_convert, id_dict[dt])
        p.close()
        p.join()

    # t2=time.time()
    # print('所用时间：',t2-t1)
