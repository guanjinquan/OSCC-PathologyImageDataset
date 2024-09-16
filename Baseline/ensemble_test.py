from datasets import GetDataLoader
from models import GetModel
from utils import load_model, parse_arguments
import numpy as np
from utils.config import parse_arguments
import torch
import random
import os
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

def show_metrics(y_prob, y_true):
    y_pred = [0 if i < 0.5 else 1 for i in y_prob]  # 二分类取最大值
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_prob)
    return {"acc": round(accuracy, 4), "recall": round(recall, 4), "f1": round(f1, 4), "precision": round(precision, 4), "auc": round(auc_value, 4)}


class EnsembleTester:
    def __init__(self, fold=0, args=None):  
        self.args = parse_arguments() if args is None else args  # args可能是None，得用self.args
        self.fold = fold
        
        if self.args.load_pth_path is None:
            task = eval(self.args.use_tasks)[0]
            if fold == 0:
                load_pth_path = os.path.join(self.args.ckpt_path, self.args.model, self.args.runs_id, f"valid_{task}_Best.pth")
            else:
                load_pth_path = os.path.join(self.args.ckpt_path, self.args.model, self.args.runs_id, f"fold{fold}", f"valid_{task}_Best.pth")
        else:
            load_pth_path = self.args.load_pth_path
        
        # dataset 
        mean_std = ([175.14728804175988, 110.57123792228117, 176.73598615775617], [21.239463551725915, 39.15991384752335, 10.99100631656543])
        self.train_loader, self.val_loader, self.test_loader = \
            GetDataLoader(fold, mean_std, self.args, True)
        
        # running setting
        self.model = GetModel(self.args).cuda()
        assert load_pth_path is not None, "load_path can't be None."
        print(f"Load from {load_pth_path}!!!", flush=True)
        cp = load_model(load_pth_path)
        pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
        self.model.load_state_dict(pretrain)

    
    def run(self):
        ret = {}
        if self.val_loader is not None:
            ret['valid'] = self.eval_epoch(self.val_loader, 'valid')
        if self.test_loader is not None:
            ret['test'] = self.eval_epoch(self.test_loader, 'test')
        return ret
    
    def eval_epoch(self, val_loader, mode='valid'):
        self.model.eval()
        with torch.no_grad():
            loss = {}
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
                
                _, losses = self.model.loss(out, y) 
                for k, v in losses.items():
                    loss[k] = loss.get(k, []) + [v]
                for k, v in out.items():
                    outs[k] = outs.get(k, [])
                    outs[k].extend(v.detach().cpu().numpy().tolist())
                for k, v in y.items():
                    true[k] = true.get(k, [])
                    true[k].extend(v.detach().cpu().numpy().tolist())
                
            loss_dict, metrics_dict = {}, {}
            for k, v in loss.items():
                loss_dict[f"loss_{k}_{mode}"] = np.mean(v)
            all_metrics = self.model.metrics(outs, true)
            for k, metrics in all_metrics.items():
                for m, a in metrics.items():
                    metrics_dict[f"{m}_{k}_{mode}"] = a

            print(f"{self.args.model} : {all_metrics}")
            
            return {"pids": pids, "outs": outs, "true": true}

"""
python /mnt/home/Guanjq/BackupWork/PathoCls/Baseline/ensemble_test.py \
    --gpu_id "1" \
    --seed 109 \
    --batch_size 2 \
    --split_filename "split_final_seed=2024.json" \
    --datainfo_file "all_pathology_info.json" \
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
        'vit_base_p16_hibou': ["008_vit_base_p16_hibou", ],
        "vit_small_p16_pathology": ["002_LNM_vit_bench", "017_LNM_vit_bench"],
        "swin_imagenet": ["004_swin_imagenet"]
        # "densenet121_imagenet": "006_densenet121_imagenet"
    }
    
    # start work
    ret = {}
    for model, runs_ids in ensemble_dict.items():
        args.model = model
        for runs_id in runs_ids:
            args.runs_id = runs_id
            trainer = EnsembleTester(fold=0, args=args)
            ret[runs_id] = trainer.run()
        
    with open(f"ensemble.json", "w") as f:
        json.dump(ret, f)
        
    # clinic_paths = [
    #     "/mnt/home/Guanjq/BackupWork/PathoCls/ClinicResults/test_result.json"
    # ]
    
    # for cp in clinic_paths:
    #     with open(cp, "r") as f:
    #         clinic_res = json.load(f)
    #     ret[cp] = {}
    #     for mode in ['valid', 'test']:
    #         ret[cp][mode] = {"outs": {"LNM": []}}
    #         for prob in clinic_res[mode]['outs']:
    #             ret[cp][mode]["outs"]["LNM"].append([0, prob])
            

    task = eval(args.use_tasks)[0]
    model = GetModel(args)
    for mode in ['valid', 'test']:
        max_probs = []
        min_probs = []
        mean_probs = []
        vote_probs = []
        trues = list(ret.values())[0][mode]['true'][task]
        
        # print(len(trues))
        
        # max 融合
        for i in range(len(trues)):
            prob = 0
            for k in ret.keys():
                prob = max(prob, ret[k][mode]['outs'][task][i][1])
            max_probs.append(prob)
        
        # min 融合
        for i in range(len(trues)):
            prob = 1
            for k in ret.keys():
                prob = min(prob, ret[k][mode]['outs'][task][i][1])
            min_probs.append(prob)
        
        # mean 融合
        for i in range(len(trues)):
            prob = 0
            for k in ret.keys():
                prob += ret[k][mode]['outs'][task][i][1]
            mean_probs.append(prob / len(ret.keys()))
        
        # vote 融合
        for i in range(len(trues)):
            prob = 0
            probs = []
            for k in ret.keys():
                probs.append(ret[k][mode]['outs'][task][i][1])
                if ret[k][mode]['outs'][task][i][1] >= 0.5:
                    prob += 1
            if prob >= len(ret.keys()) / 2:
                vote_probs.append(max(probs))
            else:
                vote_probs.append(min(probs))
        
        # print(len(mean_probs), len(trues))
        
        mean_metrics = show_metrics(mean_probs, trues)
        max_metrics = show_metrics(max_probs, trues)
        min_metrics = show_metrics(min_probs, trues)
        vote_metrics = show_metrics(vote_probs, trues)
        
        print(f"{mode} mean_metrics: {mean_metrics}")
        print(f"{mode} max_metrics: {max_metrics}")
        print(f"{mode} min_metrics: {min_metrics}")
        print(f"{mode} vote_metrics: {vote_metrics}")
    
    
    