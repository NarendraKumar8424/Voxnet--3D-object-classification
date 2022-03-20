# coding: utf-8
import os
#import sys
import numpy as np
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
#sys.path.insert(0, '../utils/')
#sys.path.insert(0, './utils/')
import binvox_rw


class MCB_B(Dataset):
    def __init__(self, data_root, n_classes, idx2cls, split='train'):
        """
        Args:
            split (str, optional): 'train' or 'test'. Defaults to 'train'.
        """
        self.data_root = data_root
        self.n_classes = n_classes
        self.samples_str = []
        self.cls2idx = {}
        for k, v in idx2cls.items():
            self.cls2idx.update({v: k})
            for sample_str in glob.glob(os.path.join(data_root, v, split, '*.binvox')):
                if re.match(r"\d+.binvox", os.path.basename(sample_str)):
                    self.samples_str.append(sample_str)
        print(self.cls2idx)

    def __getitem__(self, idx):
        sample_name = self.samples_str[idx]
        cls_name = re.split(r"/", sample_name)[-3]
        cls_idx = self.cls2idx[cls_name]
        with open(sample_name, 'rb') as file:
            data = np.int32(binvox_rw.read_as_3d_array(file).data)
            data = data[np.newaxis, :]

        sample = {'voxel': data, 'cls_idx': cls_idx}

        return sample

    def __len__(self):
        return len(self.samples_str)


if __name__ == "__main__":
    idx2cls = {0: 'bearing', 1: 'bushing', 2: 'castors_and_wheels', 3: 'clamp', 4: 'disc',
               5: 'fitting', 6: 'flange', 7: 'fork_joint', 8: 'gear', 9: 'handles', 
               10: 'hinge', 11: 'hook', 12: 'motor', 13: 'nut', 14: 'pin', 15: 'plate',
               16: 'pulley', 17: 'ring', 18: 'rivet', 19: 'rotor', 20: 'screws_and_bolts', 
               21: 'spring', 22: 'stud', 23: 'switch', 24: 'washer'}

    data_root = '/home/yelisett/MCB_B'

    dataset = MCB_B(data_root=data_root, n_classes=25, idx2cls=idx2cls, split='train')
   
    cnt = len(dataset)
    

    data, cls_idx = dataset[0]['voxel'], dataset[1]['cls_idx']
    print(f"length: {cnt}\nsample data: {data}\nsample cls: {cls_idx}")
