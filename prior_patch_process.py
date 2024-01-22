from utils.utils import get_csv_split
from utils.Get_patch_based_centerline import Get_patch_from_pre
import yaml
import multiprocessing
import argparse
import os

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--Direct_model',type=str,default='FCN')
    p.add_argument('--Direct_parameter',type=str,default='Mid_resolution_4_Dice')
    p.add_argument('--pools',type=int,default=4)

    args = p.parse_args()
    k = args.fold
    config_file = args.config_file
    coarse_version=args.Direct_model
    direct_parameters =args.Direct_parameter
    pool_num=args.pools

    print('coarse_version:',coarse_version)

    # 根据预分割进行裁剪
    with open(config_file) as f:
        config = yaml.load(f)

    img_path = config['General_parameters']['data_path']
    csv_path = config['General_parameters']['csv_path']
    mid_path = config['General_parameters']['mid_path']

    patch_path = os.path.join(mid_path,'Prior_Patches',coarse_version,direct_parameters,'fold_%d'% k)
    p_path = os.path.join('result/Direct_seg',coarse_version,direct_parameters,'fold_%d' % k, 'pre_label')

    id_dict = get_csv_split(csv_path, k)

    # 获取体素块
    for p_size in [64,32,16]:
        for dt in ['train', 'valid']:
            print('get_patch %s %d' % (dt, p_size))
            get_patch_opt = Get_patch_from_pre(img_path, img_path, p_path, patch_path, p_size, dt)
            p = multiprocessing.Pool(pool_num)
            p.map(get_patch_opt.run, id_dict[dt])
            p.close()
            p.join()
