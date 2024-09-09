import sys
sys.path.append("/home/Guanjq/Work/MultiSitePathologyReccurence/Code")
from torch.utils.data import Dataset, sampler
import random


class BalancedBatchSampler(sampler.Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        labels = dataset._get_labels()

        self.dataset = dict()     # {'label0':[ 属于label0的下标], 'label1': [...]} 
        self.balanced_max = 0     # 所有label中最大的数量
        
        # 将所有类别的索引保存在字典self.dataset中
        for idx in range(0, len(dataset)):
            key = labels[idx]
            if key not in self.dataset:
                self.dataset[key] = list()
            self.dataset[key].append(idx)
            # 记录类别中样本最多的数量，以便后面进行平衡采样
            self.balanced_max = len(self.dataset[key]) \
                if len(self.dataset[key]) > self.balanced_max else self.balanced_max

        # 对样本数少于最多样本数的类别进行过采样
        for key in self.dataset.keys():
            while len(self.dataset[key]) < self.balanced_max:
                self.dataset[key].append(random.choice(self.dataset[key]))
        
        self.keys = list(self.dataset.keys()) # list(range(class_nums))  
        self.currentkey_idx = 0
        self.indices = self._init_indices()
    
    def _init_indices(self):
        indices = dict()
        for key in self.keys:
            indices[key] = -1
        return indices
    
    def __iter__(self):# -> Generator[Any, Any, None]:
        while self.indices[self.keys[self.currentkey_idx]] < self.balanced_max - 1:
            self.indices[self.keys[self.currentkey_idx]] += 1
            yield self.dataset[self.keys[self.currentkey_idx]][self.indices[self.keys[self.currentkey_idx]]]    # 每一次只返回一个下标
            self.currentkey_idx = (self.currentkey_idx + 1) % len(self.keys)
        self.indices = self._init_indices()

    def __len__(self):
        # 返回平衡采样后的总样本数量
        return self.balanced_max * len(self.keys)