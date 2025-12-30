import os
import sys
sys.path.append(os.path.dirname(__file__) + '/../Baseline')
sys.path.append(os.path.dirname(__file__) + '/../')
import torch.nn as nn
import cv2
import numpy as np
import torch
from Baseline.models import GetModel
from Baseline.oldmodels import GetModel as GetOldModel
from Baseline.datasets import GetDataLoader
from Baseline.utils import parse_arguments, load_model
import random
import glob
from collections import defaultdict
import itertools


class TaskSpecificModel(nn.Module):
    def __init__(self, task, model):
        super(TaskSpecificModel, self).__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        x = self.model(x)
        return x[self.task]



def work(task, pth_path, pid_list):
    args = parse_arguments()
    args.batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    args.gpu_id = gpu_id
    
    args.split_filename = "split_seed=2024.json"
    args.datainfo_file = "all_metadata.json"
    
    basename = os.path.basename(pth_path)
    args.model = basename.split('-')[1].split('.')[0]
    task = basename.split('-')[0]
    args.use_tasks = f"['{task}']"
    
    # fixed seed
    seed = 17
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # dataset 
    mean_std = ([175.14728804175988, 110.57123792228117, 176.73598615775617], [21.239463551725915, 39.15991384752335, 10.99100631656543])
    train_loader, val_loader, test_loader = GetDataLoader(0, mean_std, args, True)
        
    
    # running setting
    model = GetModel(args).cuda()
    assert pth_path is not None, "load_path can't be None."
    print(f"Load from {pth_path}!!!", flush=True)
    cp = load_model(pth_path)
    
    flag = False
    for k, v in cp['model'].items():
        if 'fusion_block' in k:
            flag = True
            break
    
    if flag:
        pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
        pretrain = {k: v for k, v in pretrain.items() if k in model.state_dict()}
        model.load_state_dict(pretrain)
    else:
        model = GetOldModel(args).cuda()
        pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
        pretrain = {k: v for k, v in pretrain.items() if k in model.state_dict()}
        model.load_state_dict(pretrain)
    
    model = TaskSpecificModel(task, model).cuda()

    # loader = train_loader
    loader = itertools.chain(train_loader, val_loader, test_loader)
    for x, y, ids in loader:
        
        if int(ids[0].item()) not in pid_list:
            continue
        
        x = x.cuda()
        for k, v in y.items():
            y[k] = v.cuda()
        
        label = y[task].item()
        
        model.eval()
        with torch.no_grad():
            out = model(x)
        probs = torch.softmax(out, dim=1)
        if torch.argmax(probs, dim=1) != label:
            print(f"Skip {ids} due to wrong prediction.", flush=True)
            continue
        else:
            print("Prediction Correct!!!", flush=True)


if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    gpu_id = "1"

    task_pidlist = {
        "REC": [868852],
        "LNM": [868852],
        "TD": [868852],
        "TI": [868852],
        "CE": [868852],
        "PI": [868852]
    }

    
    # setting config
    origin_images_path = "/home/Guanjq/HuangData/PathologyImages/"
    # load_pth_path = "./BestCheckpoints/TI-vit_small_p16_pathology.pth"
    # load_pth_path = "./BestCheckpoints/REC-vit_base_imagenet.pth"
    # load_pth_path = "./BestCheckpoints/REC-vit_small_p16_pathology.pth"
    # load_pth_path = "./BestCheckpoints/CE-vit_small_p16_pathology.pth"
    # load_pth_path = "./BestCheckpoints/LNM-vit_small_p16_pathology.pth"
    # load_pth_path = "./BestCheckpoints/TD-vit_small_p16_pathology-reinhard.pth"
    # load_pth_path = "./BestCheckpoints/TD-vit_base_p16_conch.pth"
    # load_pth_path = "./BestCheckpoints/PI-vit_small_p16_pathology-reinhard.pth"
    # load_pth_path = "./BestCheckpoints/PI-vit_base_p16_conch.pth"
    
    task_pth = {
        "REC": "./BestCheckpoints/REC-vit_small_p16_pathology.pth",
        "LNM": "./BestCheckpoints/LNM-vit_small_p16_pathology.pth",
        "TD": "./BestCheckpoints/TD-vit_small_p16_pathology-reinhard.pth",
        "TI": "./BestCheckpoints/TI-vit_small_p16_pathology.pth",
        "CE": "./BestCheckpoints/CE-vit_small_p16_pathology.pth",
        "PI": "./BestCheckpoints/PI-vit_small_p16_pathology-reinhard.pth"
    }
    
    # inference setting
    os.chdir(os.path.dirname(__file__) + "/../")
    
    for task, pid_list in task_pidlist.items():
        work(task, task_pth[task], pid_list)
    
    