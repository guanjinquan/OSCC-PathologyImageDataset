from datasets import GetDataLoader
from models import GetModel
from oldmodels import GetModel as GetOldModel
from utils import load_model, parse_arguments
from sklearn.metrics import roc_auc_score
import numpy as np
from utils.config import parse_arguments
import torch
import random
import os
import json


def bootstrap_auc(labels, probs, num_classes, bootstraps = 100, fold_size = 200):
    statistics = np.zeros((num_classes, bootstraps))

    for c in range(num_classes):
        pos_probs = probs[np.where(labels == c), c].reshape(-1)
        neg_probs = probs[np.where(labels != c), c].reshape(-1)
        prevalence = len(pos_probs) / len(labels)
        for i in range(bootstraps):
            pos_sample = np.random.choice(pos_probs, int(fold_size * prevalence), replace=True)
            neg_sample = np.random.choice(neg_probs, int(fold_size * (1-prevalence)), replace=True)

            labels_sample = np.concatenate([np.ones(len(pos_sample)), np.zeros(len(neg_sample))])
            probs_sample = np.concatenate([pos_sample, neg_sample])
            score = roc_auc_score(labels_sample, probs_sample)
            statistics[c][i] = score
    return statistics

def roc_auc_confidence_interval(statistics, alpha = 0.95):
    lower = (1 - alpha) / 2
    upper = alpha + lower
    lower_bound = np.quantile(statistics, lower, axis = 0)
    upper_bound = np.quantile(statistics, upper, axis = 0)
    return lower_bound, upper_bound


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
        # if self.val_loader is not None:
        #     ret['valid'] = self.eval_epoch(self.val_loader, 'valid')
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
                    loss[k] = loss.get(k, []) + [v.item()]
                for k, v in out.items():
                    outs[k] = outs.get(k, []) + v.detach().cpu().numpy().tolist()
                for k, v in y.items():
                    true[k] = true.get(k, []) + v.detach().cpu().numpy().tolist()
                
            loss_dict, metrics_dict = {}, {}
            for k, v in loss.items():
                loss_dict[f"loss_{k}_{mode}"] = np.mean(v)
            all_metrics = self.model.metrics(outs, true)
            for k, metrics in all_metrics.items():
                for m, a in metrics.items():
                    metrics_dict[f"{m}_{k}_{mode}"] = a
                num_classes = len(set(true[k]))
                probs = np.array(outs[k])
                labels = np.array(true[k])
                statistics = bootstrap_auc(labels, probs, num_classes)
                metrics_dict[f"95AUC_CI_{k}_{mode}"] = tuple(list(np.round(np.mean([
                    roc_auc_confidence_interval(statistics[i]) for i in range(num_classes)
                ], axis=0), 4)))
                temp_key0 = f"AUC_{k}_{mode}"
                temp_key1 = f"95AUC_CI_{k}_{mode}"
                metrics_dict[f"log_{k}_{mode}"] = f"\\makecell{{{round(100*metrics_dict[temp_key0], 2)} // {(round(100*metrics_dict[temp_key1][0], 2), round(100*metrics_dict[temp_key1][1], 2))}}}"
                
            print(f"{mode} : {loss_dict}")
            print(f'metrics : ' + str(metrics_dict))
            return {"pids": pids, "outs": outs, "true": true}

    
if __name__ == '__main__':
    args = parse_arguments()
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
    
    
    # start work
    if args.train_mode == "TVT":
        trainer = Tester(fold=0, args=args)
        trainer.run()
    else:
        cv_list = eval(args.train_mode)
        error_ids = {}
        task = eval(args.use_tasks)[0]
        for iter_fold in cv_list:
            print("Woriking on fold " + str(iter_fold))
            error_ids[iter_fold] = []
            trainer = Tester(fold=iter_fold, args=args)
            ret_res = trainer.run()
            valid_res = ret_res['valid']
            for i in range(len(valid_res['pids'])):
                pred = int(valid_res['outs'][task][i][1] >= 0.5)
                if pred != valid_res['true'][task][i]:
                    error_ids[iter_fold].append(valid_res['pids'][i])
        with open(f"error_ids_{task}.json", 'w') as f:
            json.dump(error_ids, f)
