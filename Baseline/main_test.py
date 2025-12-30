from datasets import GetDataLoader
from models import GetModel
from oldmodels import GetModel as GetOldModel
from utils import load_model, parse_arguments
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from utils.config import parse_arguments
import torch
import random
import os
import json


# 这个函数本身并不直接输出 95 % 置信区间，它只返回了每个类别下 bootstraps 次重复抽样得到的 AUC 值分布 statistics[c]。
# 需要调用confidence_interval进一步获取
def bootstrap_auc(labels, probs, num_classes, bootstraps = 500):
    state = np.random.get_state()
    np.random.seed(2024)  # for reproducibility
    statistics = np.zeros((num_classes, bootstraps))
    labels = np.array(labels)
    probs = np.array(probs)
    fold_size = labels.shape[0]
    print("fold_size = ", fold_size)
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
    np.random.set_state(state)
    return statistics


def bootstrap_CI(func_method, labels, preds, num_classes, bootstraps = 500):
    state = np.random.get_state()
    np.random.seed(2024)  # for reproducibility
    statistics = np.zeros((1, bootstraps))
    
    labels = np.array(labels)
    preds = np.array(preds)
    sample = [[] for _ in range(num_classes)]
    for i in range(labels.shape[0]):
        sample[labels[i]].append(preds[i])
        
    for i in range(bootstraps):
        # balance each class
        sample_preds = []
        sample_labels = []
        for c in range(num_classes):
            temp_preds = np.random.choice(sample[c], len(sample[c]), replace=True)
            sample_labels += [c] * len(temp_preds)
            sample_preds += temp_preds.tolist()
        if (func_method.__name__ != 'accuracy_score'):
            score = func_method(sample_labels, sample_preds, average='macro')
        else:
            score = func_method(sample_labels, sample_preds)
        statistics[0][i] = score
    
    np.random.set_state(state)
    return statistics

def confidence_interval(statistics, alpha = 0.95):
    lower = (1 - alpha) / 2
    upper = alpha + lower
    lower_bound = np.quantile(statistics, lower, axis = 0)
    upper_bound = np.quantile(statistics, upper, axis = 0)
    return lower_bound, upper_bound


def Box_Plot(statistics):
    lowest = np.min(statistics, axis=0)
    highest = np.max(statistics, axis=0)
    Q2 = np.quantile(statistics, 0.5, axis=0)
    Q1 = np.quantile(statistics, 0.25, axis=0)
    Q3 = np.quantile(statistics, 0.75, axis=0)
    return lowest, Q1, Q2, Q3, highest


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
            func_methods = {
                "AUC": roc_auc_score,
                "Acc": accuracy_score,
                "F1": f1_score,
                "Precision": precision_score,
                "Recall": recall_score
            }
            for k, metrics in all_metrics.items():
                probs = torch.softmax(torch.tensor(outs[k]), dim=1).numpy()
                preds = np.argmax(probs, axis=1).astype(np.int32)
                labels = true[k]
                num_classes = len(set(true[k]))
                for m, a in metrics.items():
                    metrics_dict[f"{m}_{k}_{mode}"] = a
                    if m in list(func_methods.keys()):
                        if m == "AUC":
                            statistics = bootstrap_auc(labels, probs, num_classes)
                            metrics_dict[f"95CI_{m}_{k}_{mode}"] = tuple(list(np.round(np.mean([
                                confidence_interval(statistics[i]) for i in range(num_classes)
                            ], axis=0), 4)))
                            metrics_dict[f"BOX_{m}_{k}_{mode}"] = tuple(list(np.round(np.mean([
                                Box_Plot(statistics[i]) for i in range(num_classes)
                            ], axis=0), 4)))

                        else:
                            statistics = bootstrap_CI(func_methods[m], labels, preds, num_classes)
                            metrics_dict[f"95CI_{m}_{k}_{mode}"] = tuple(list(
                                np.round(confidence_interval(statistics[0]), 4)
                            ))
                            
                print(self.model.tasks[k].metrics(torch.tensor(outs[k]), true[k]))
                    

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
