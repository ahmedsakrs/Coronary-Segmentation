import multiprocessing
import numpy as np
import os
import yaml
from utils.utils import get_csv_split
from utils.Make_graph import Make_Graph
import argparse

if __name__=='__main__':

    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--Direct_model', type=str, default='FCN')
    p.add_argument('--Direct_parameter', type=str, default='Mid_resolution_4_Dice')
    p.add_argument('--pools', type=int, default=4)

    args = p.parse_args()
    k = args.fold
    config_file = args.config_file
    coarse_version = args.Direct_model
    direct_parameters = args.Direct_parameter
    pool_num = args.pools

    print('coarse_version: %s || fold:%d'%(coarse_version,k))

    with open(r'config/config.yaml') as f:
        config = yaml.load(f)

    img_path = config['General_parameters']['data_path']
    csv_path = config['General_parameters']['csv_path']
    mid_path = config['General_parameters']['mid_path']

    graph_path=os.path.join(mid_path, 'Graph', coarse_version, direct_parameters, 'fold_%d' % k,'graph')
    pre_seg_path = os.path.join('result/Direct_seg', coarse_version, direct_parameters, 'fold_%d' % k, 'pre_label')
    id_dict=get_csv_split(csv_path,k)

    for dt in ['train','valid']:
        make_graph_opt = Make_Graph(img_path, img_path, pre_seg_path, graph_path,dt)
        print('convert_tree %s' % dt)
        p=multiprocessing.Pool(pool_num)
        p.map(make_graph_opt.run,id_dict[dt])
        p.close()
        p.join()
