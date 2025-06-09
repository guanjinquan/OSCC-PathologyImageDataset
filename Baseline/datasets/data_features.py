from torch.utils.data import Dataset
import numpy as np
import random
import os
import json
from datasets.default import default_augment
import tqdm

class MyFeatDataset(Dataset):
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
        feats = np.load(self.items[index]['path']).astype(np.uint8)
        
        labels = {}
        for task in eval(self.args.use_tasks):
            labels[task] = self.items[index][task]
        
        if self.data_type == "CORE":
            feats = feats[0:3]  
        elif self.data_type == "EDGE":
            feats = feats[3:6]  
            
        if self.args.num_feat != -1:
            TARGET_NUM = self.args.num_feat
        else:
            TARGET_NUM = 3 if self.data_type in ["EDGE", "CORE"] else 6
        assert feats.shape[0] == TARGET_NUM, f"error = {self.items[index]['pid']}"
        
        return [feats, labels, self.items[index]['pid']]  # feature, labels, patient_id
    
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
    

    