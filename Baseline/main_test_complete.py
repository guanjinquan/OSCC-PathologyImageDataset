from datasets import GetDataLoader
from models import GetModel
from oldmodels import GetModel as GetOldModel
from utils import load_model, parse_arguments
import numpy as np
from utils.config import parse_arguments
import torch
import random
import os
import json

class Tester:
    def __init__(self, fold=0, args=None):  
        self.args = parse_arguments() if args is None else args  # args可能是None，得用self.args
        self.fold = fold
        
        
        tasks = eval(self.args.use_tasks)
        if len(tasks) > 1:
            if fold == 0:
                load_pth_path = os.path.join(self.args.ckpt_path, self.args.model, self.args.runs_id, f"valid_MultiTask_Best.pth")
            else:
                load_pth_path = os.path.join(self.args.ckpt_path, self.args.model, self.args.runs_id, f"fold{fold}", f"valid_MultiTask_Best.pth")
        else:
            task = tasks[0]
            if fold == 0:
                load_pth_path = os.path.join(self.args.ckpt_path, self.args.model, self.args.runs_id, f"valid_{task}_Best.pth")
            else:
                load_pth_path = os.path.join(self.args.ckpt_path, self.args.model, self.args.runs_id, f"fold{fold}", f"valid_{task}_Best.pth")
        
        # dataset 
        mean_std = ([175.14728804175988, 110.57123792228117, 176.73598615775617], [21.239463551725915, 39.15991384752335, 10.99100631656543])
        self.train_loader, self.val_loader, self.test_loader = \
            GetDataLoader(fold, mean_std, self.args, True)
        print("MeanStd = " + str(self.train_loader.dataset.mean_std))
        
        # running setting
        self.model = GetModel(self.args).cuda()
        assert load_pth_path is not None, "load_path can't be None."
        print(f"Load from {load_pth_path}!!!", flush=True)
        cp = load_model(load_pth_path)
        
        flag = False
        for k, v in cp['model'].items():
            if 'fusion_block' in k:
                flag = True
                break
        
        if flag:
            pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
            pretrain = {k: v for k, v in pretrain.items() if k in self.model.state_dict()}
            self.model.load_state_dict(pretrain)
        else:
            self.model = GetOldModel(self.args).cuda()
            pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
            pretrain = {k: v for k, v in pretrain.items() if k in self.model.state_dict()}
            self.model.load_state_dict(pretrain)

    
    def run(self):
        ret = {}
        ret['train'] = self.eval_epoch(self.train_loader, 'train')
        return ret
    
    def eval_epoch(self, val_loader, mode='valid'):
        self.model.eval()
        with torch.no_grad():
            outs = {}
            true = {}
            pids = []
            for i, (x, y, p_idxs) in enumerate(val_loader, 1):
                x = x.cuda()
                for k, v in y.items():
                    y[k] = v.cuda()
                p_idxs = p_idxs.cpu().data.numpy()
                pids.extend(p_idxs.tolist())
                
                out = self.model(x)

                
                for k, v in out.items():
                    outs[k] = outs.get(k, []) + torch.softmax(v, dim=1).detach().cpu().numpy().tolist()
                for k, v in y.items():
                    true[k] = true.get(k, []) + v.detach().cpu().numpy().tolist()
            
            return {"pids": pids, "outs": outs, "true": true}

    

"""
python /home/Guanjq/Work/OSCC-PathologyImageDataset/Baseline/main_test_complete.py \
    --gpu_id "1" \
    --seed 109 \
    --batch_size 2 \
    --split_filename "split_seed=2024.json" \
    --datainfo_file "all_metadata.json" \
    --img_size 512 \
    --use_tasks "['LNM']" 

""" 
if __name__ == '__main__':
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    os.chdir(os.path.dirname(__file__) + "/../")
    
    # 固定种子
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    ensemble_dict = {
        # 'vit_base_p16_hibou': ["008_vit_base_p16_hibou", ],
        # "vit_small_p16_pathology": ["002_LNM_vit_bench", "017_LNM_vit_bench"],
        "swin_imagenet": ["007_LNM_swin_imagenet"]
        # "densenet121_imagenet": "006_densenet121_imagenet"
    }
    
    # start work
    ret = {}
    for model, runs_ids in ensemble_dict.items():
        args.model = model
        for runs_id in runs_ids:
            args.runs_id = runs_id
            trainer = Tester(fold=0, args=args)
            ret[runs_id] = trainer.run()
    
    for k, v in ret.items():
        outs = v['train']['outs']
        true = v['train']['true']
        pids = v['train']['pids']
        
        for i in range(len(pids)):
            if true['LNM'][i] == -1:
                print(pids[i], outs['LNM'][i], true['LNM'][i])