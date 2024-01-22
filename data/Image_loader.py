from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import os
import pandas as pd
from monai.transforms import RandFlip, RandRotate
from utils.utils import reshape_img,normalize


class CoronaryImage(Dataset):
    def __init__(self, data_dir, label_dir, ID_list, img_size, transform=None, is_normal=True):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.ID_list = ID_list

        self.transform = transform
        # self.data_list = os.listdir(data_dir)
        self.output_size = img_size
        self.is_normal = is_normal

    def __len__(self):
        return len(self.ID_list)

    def __getitem__(self, index):
        image_index = self.ID_list[index]

        image_path = os.path.join(self.data_dir, image_index, 'img.nii.gz')
        label_path = os.path.join(self.label_dir, image_index, 'label.nii.gz')

        ID = image_index

        img_nii = nib.load(image_path)
        img = img_nii.get_data()
        label_nii = nib.load(label_path)
        label = label_nii.get_data()
        img_size = np.array(img.shape)

        img = reshape_img(img, self.output_size)
        label = reshape_img(label, self.output_size)

        img = np.expand_dims(img, 0)
        if self.is_normal == True:
            img = normalize(img)
        label = np.expand_dims(label, 0)
        sample = {'image': img, 'label': label, 'affine': img_nii.affine, 'image_index': ID, 'image_size': img_size}
        return sample
