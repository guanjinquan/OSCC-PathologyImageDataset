from datasets.default.default_dataset import MyBaseDataset
import numpy as np
import random
import os
import torch
from datasets.stain import stain_augment
import pickle as pkl


class StainDataset(MyBaseDataset):
    def __init__(self, stain_method, fold=0, type="train", data_type="ALL", mean_std=None, test_mode=False, args=None):
        super().__init__(fold, type, data_type, mean_std, test_mode, args)
        self.transforms = \
            stain_augment.TestTransforms(mean_std) if test_mode \
            else stain_augment.TrainTransforms(mean_std, args) 

        self.stain_dir = "./Data/StainNormedData"
        self.stain_method = stain_method
        

    def __getitem__(self, index):
        # 如果有task的label为-1，只需要在loss函数的地方乘以0即可
        labels = {}
        for task in eval(self.args.use_tasks):
            labels[task] = self.items[index][task]
        
        if self.data_type == "CORE":
            image = image[0:3]  # 只取前三张
        elif self.data_type == "EDGE":
            image = image[3:6]  # 只取后三张
        
        pat_dir = os.path.join(self.stain_dir, self.stain_method, str(self.items[index]['pid']))
        path_exist = os.path.exists(pat_dir)
        if self.type == 'train' and np.random.rand() < self.args.stain_prob and path_exist:  # 0.5 prob to use stain
            norm_img = np.random.choice(os.listdir(pat_dir))
            norm_img = os.path.join(pat_dir, norm_img)
            norm_img = np.load(norm_img)
            image = torch.from_numpy(norm_img).float()
            data = self.transforms(image)
        else:
            image = np.load(self.items[index]['path']).astype(np.uint8)
            image = torch.from_numpy(image).float()
            data = self.transforms(image)
            
        assert image.shape[0] == 6, f"error = {self.items[index]['pid']}"
        assert image.shape[1] == 3 and image.shape[2] == self.args.img_size and image.shape[3] == self.args.img_size, f"Invalid Shape : {image.shape} but config's img_size = {self.args.img_size}."
        
        return [data, labels, self.items[index]['pid']]  # image, labels, patient_id
 