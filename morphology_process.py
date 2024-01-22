import nibabel as nib
import pandas as pd
from scipy.ndimage.interpolation import zoom
import os
from utils.utils import get_csv_split
import multiprocessing
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label
from skimage.morphology import skeletonize
import argparse
import yaml
from utils.Calculate_metrics import get_region_num


class Morphology_process:

    def __init__(self, pre_path,save_path, str_key,se_size=5,con_num=2):
        self.pre_path = pre_path
        self.str_key = str_key
        self.se_size=se_size
        self.con_num=con_num
        self.save_path=save_path

    def process(self, id):
        pre_nii = nib.load(os.path.join(self.pre_path, id, self.str_key))
        pre_label = pre_nii.get_fdata()

        se = generate_sphere3d(self.se_size)
        d_img = binary_dilation(pre_label, se)
        d_img = get_region_num(d_img, self.con_num)
        cl_img = skeletonize(d_img.astype(np.uint8))

        nib.save(nib.Nifti1Image(d_img.astype(np.float), pre_nii.affine),
                 os.path.join(self.save_path, id, 'pre_dilation.nii.gz'))
        nib.save(nib.Nifti1Image(cl_img.astype(np.float), pre_nii.affine),
                 os.path.join(self.save_path, id, 'pre_cl.nii.gz'))

        print('%s:done'%id)


def generate_sphere3d(r):
    d = 2 * r + 1
    x, y, z = np.meshgrid(np.arange(d), np.arange(d), np.arange(d), indexing='ij')
    d_c = np.sqrt((x - r) ** 2 + (y - r) ** 2 + (z - r) ** 2)
    S = np.zeros_like(d_c)
    S[x[d_c <= r], y[d_c <= r], z[d_c <= r]] = 1
    return S


if __name__=='__main__':

    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--config_file', type=str, default='config/config.yaml')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--Direct_model',type=str,default='FCN')
    p.add_argument('--Direct_parameter',type=str,default='Mid_resolution_4_Dice')
    p.add_argument('--pools',type=int,default=16)

    args = p.parse_args()
    k = args.fold
    config_file = args.config_file
    coarse_version=args.Direct_model
    direct_parameters =args.Direct_parameter
    pool_num=args.pools

    with open(config_file) as f:
        config = yaml.load(f)

    p_path = os.path.join('result/Direct_seg',coarse_version,direct_parameters, 'fold_%d' % k, 'pre_label')
    img_path = config['General_parameters']['data_path']
    csv_path = config['General_parameters']['csv_path']

    # save_label_path = r'result/%s/fold_%d/pre_label' % (version, k)
    print('Morphology process......')
    mp=Morphology_process(p_path,p_path,'pre_label.nii.gz')

    ID_dict=get_csv_split(csv_path,k)
    ID_list=ID_dict['valid']+ID_dict['train']
    p = multiprocessing.Pool(pool_num)
    p.map(mp.process,ID_list)
    p.close()
    p.join()
