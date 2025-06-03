from torch.utils.data import Dataset
import numpy as np
import random
import os
import json
from datasets.default import default_augment
from PIL import Image
import tqdm

class MyBaseDataset(Dataset):
    def __init__(self, fold=0, type="train", data_type="ALL", mean_std=None, test_mode=False, args=None):
        super().__init__()
        assert test_mode is not None, "test_mode can't be None."
        assert args is not None, "args can't be None."
        
        # 所有成员变量归纳在这里
        self.args = args
        self.mean_std = mean_std
        self.data_type = data_type
        self.type = type
        self.items = []
        self.fold = fold
        self.transforms = None
        
        # 计算items
        self._load_items()
        
        # 计算mean_std
        if self.mean_std is None:
            assert self.type == "train", "Mean_Std can't be None when type is not train."
            self.mean_std = self._get_mean_std()
            
        # 确保mean_std正常之后
        self.transforms = \
            default_augment.TestTransforms(self.mean_std) if test_mode else \
            default_augment.TrainTransforms(self.mean_std, self.args)   
        
        # 如果是train模式，random dataset
        if type == "train":
            for idx in range(1, len(self.items)):  # random shuffle
                idx2 = random.randint(0, idx)
                self.items[idx], self.items[idx2] = self.items[idx2], self.items[idx]
            
        # 打印数据集信息
        print(f"Dataset {self.fold} {self.type} {self.data_type} loaded. Length: {len(self.items)}")
             
    def _get_pids(self):
        pids = []
        for item in self.items:
            pids.append(item['pid'])
        return pids

    def _get_labels(self):
        ret = []
        for item in self.items:
            labels = []
            for task in eval(self.args.use_tasks):
                labels.append(item[task])
            ret.append(tuple(labels))
        return ret
    
    def __getitem__(self, index):
        # 如果有task的label为-1，只需要在loss函数的地方乘以0即可
        image = np.load(self.items[index]['path']).astype(np.uint8)
        
        labels = {}
        for task in eval(self.args.use_tasks):
            labels[task] = self.items[index][task]
        
        if self.data_type == "CORE":
            image = image[0:3]  # 只取前三张
        elif self.data_type == "EDGE":
            image = image[3:6]  # 只取后三张
        
        if self.args.only_grey:
            for i in range(image.shape[0]):
                img = image[i].transpose(1, 2, 0)
                img = np.array(Image.fromarray(img).convert('L'))
                image[i] = img.reshape(1, self.args.img_size, self.args.img_size).repeat(3, axis=0)
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        TARGET_NUM = 3 if self.data_type in ["EDGE", "CORE"] else 6
        assert image.shape[0] == TARGET_NUM, f"error = {self.items[index]['pid']}"
        if isinstance(self.args.img_size, int):
            assert image.shape[1] == 3 and image.shape[2] == self.args.img_size and image.shape[3] == self.args.img_size, f"Invalid Shape : {image.shape} but config's img_size = {self.args.img_size}."
        else:
            assert image.shape[1] == 3 and image.shape[2] == self.args.img_size[0] and image.shape[3] == self.args.img_size[1], f"Invalid Shape : {image.shape} but config's img_size = {self.args.img_size}."

        return [image, labels, self.items[index]['pid']]  # image, labels, patient_id
    
    def __len__(self):
        return len(self.items)
    
    def _load_items(self):
        datainfo_file = self.args.datainfo_file
        datainfo_path = os.path.join(self.args.data_root, datainfo_file)
        with open(datainfo_path, 'r') as f:
            self.items = json.load(f)['datainfo']

        # filter items
        split_path = os.path.join(self.args.data_root, self.args.split_filename)
        with open(split_path, 'r') as f:
            split = json.load(f)
            if self.fold > 0:
                print("Cross Validation Fold = ", self.fold)
                split = split[f"{self.fold}"]
        target_pid = set(list(map(int, split[self.type])))
        self.items = list(filter(lambda x: x['pid'] in target_pid, self.items))
        assert self._check_datapath(), "Not all paths exist!"

        # debug mode
        if self.args.debug_mode:  
            temp_items = {}
            task = eval(self.args.use_tasks)
            for item in self.items:
                key = []
                for t in task:
                    key.append(item[t])
                key = tuple(key)
                temp_items[key] = temp_items.get(key, [])
                temp_items[key].append(item)
            self.items = []
            for k, v in temp_items.items():
                self.items.extend(v[:10])
    
    
    def _check_datapath(self):
        for item in self.items:
            if not os.path.exists(item['path']):
                print("Path not exists!", item['path'], flush=True)
                print("May need to change img_size.", flush=True)
                return False
        return True
    
    # 均值方差都是由[0,255]值域的图片计算的
    def _get_mean_std(self, recalculate=False):
        if self.mean_std is not None and not recalculate:
            return self.mean_std
        means = [[], [], []]
        stds = [[], [], []]
        print("Begin to calculate mean and std.", flush=True)
        with tqdm.tqdm(total=len(self.items)) as pbar:
            for item in self.items:
                image = np.load(item['path'])  # [6, 3, self.args.img_size, self.args.img_size]
                if self.args.only_grey:
                    for i in range(image.shape[0]):
                        img = image[i].astype(np.uint8).transpose(1, 2, 0)
                        img = np.array(Image.fromarray(img).convert('L'))
                        for c in range(3):
                            means[c].append(img.mean())
                            stds[c].append(img.std())
                else:
                    for c in range(3):
                        means[c].append(image[:, c, :, :].mean())
                        stds[c].append(image[:, c, :, :].std())
                pbar.update(1)
        mean = np.array([np.mean(means[c]) for c in range(3)])
        std = np.array([np.mean(stds[c]) for c in range(3)])
        self.mean_std = (mean, std)
        return self.mean_std
        
    