from datasets import GetDataLoader
from models import GetModel
from settings import GetOptimizer, GetScheduler
from utils import Logger, save_trainer, save_model, load_model
import os
import tqdm
import numpy as np
import wandb
import torch

def get_score(metrics, key_task):
    score = 0.6 * metrics[f"AUC_{key_task}_valid"] + 0.1 * metrics[f"F1_{key_task}_valid"] + 0.3 * metrics[f"Acc_{key_task}_valid"]
    return score

        
class Trainer:
    def __init__(self, fold=0, args=None):  
        assert args is not None, 'Please input args!!!'
        self.args = args
        self.fold = fold
        
        # dataset 
        mead_std = ([175.14728804175988, 110.57123792228117, 176.73598615775617], [21.239463551725915, 39.15991384752335, 10.99100631656543])
        self.train_loader, self.val_loader, self.test_loader = \
            GetDataLoader(fold, mead_std, self.args, False)
        print("MeanStd = " + str(self.train_loader.dataset.mean_std))
        
        # running setting
        self.model = GetModel(self.args).cuda()
        if self.args.finetune:
            print(f"Fine-tune from {self.args.load_pth_path}!!!", flush=True)
            cp = load_model(self.args.load_pth_path)
            pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
            self.model.load_state_dict(pretrain)
        self.optimizer = GetOptimizer(self.args, self.model)
        self.scheduler  = GetScheduler(self.args, self.optimizer)
        self.epoch = 0
        self.best_metrics = {}
        self.best_score = 0
        self.acc_step = self.args.acc_step  # accumulate_step
        self.loss_history = []
        self.patience = self.args.num_epochs // 2
        self.monitor_length = 50  # monitor the last 50 epochs
        
        # trainer config
        run_path = [self.args.model, self.args.runs_id, 'fold'+str(fold)][:2+int(fold>0)]
        log_path = os.path.join(self.args.log_path, *run_path)
        self.ckpt_path = os.path.join(self.args.ckpt_path, *run_path) 
        print("log_path : ", log_path, flush=True)
        print("ckpt_path : ", self.ckpt_path, flush=True)
        
        if os.path.exists(os.path.join(self.ckpt_path, 'Final_Trainer.pkl')):
            print("Trainer already exists!!!", flush=True)
            raise ValueError("Trainer already exists!!!")
        
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.log = Logger(os.path.join(log_path, 'log.txt'))
        self.log.write("settings : " + str(args))
        self.log.write("MeanStd = " + str(self.train_loader.dataset.mean_std))
    
    def run(self):
        start_epoch = self.epoch
        
        for epoch_id in range(start_epoch, self.args.num_epochs + 1):  
            if epoch_id > start_epoch:  # start_epoch 不训练
                self.train_epoch(self.train_loader)
                self.scheduler.step()
                for idx, groups in enumerate(self.optimizer.param_groups):
                    wandb.log({f"lr_{['backbones', 'head'][idx]}": groups['lr']}, step=self.epoch)
            if self.val_loader is not None:
                self.eval_epoch(self.val_loader, 'valid')
            if self.test_loader is not None:
                self.eval_epoch(self.test_loader, 'test')
            self.on_epoch_end()
            # early stopping
            if epoch_id - start_epoch > self.patience and epoch_id > self.monitor_length:
                if self.loss_history[-1] > np.mean(self.loss_history[-self.monitor_length:]):
                    print(f"Early stopping at epoch {epoch_id}!!!", flush=True)
                    self.log.write(f"Early stopping at epoch {epoch_id}!!!")
                    self.log.write("Loss History : " + str(self.loss_history[-self.monitor_length:]))
                    break
                
    def train_epoch(self, train_loader):
        self.model.train()
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            loss = {}
            outs = {}
            true = {}
            self.optimizer.zero_grad()            
            for i, (x, y, p_idxs) in enumerate(train_loader, 1):
                x = x.cuda()
                for k, v in y.items():
                    y[k] = v.cuda()
                p_idxs = p_idxs.cpu().data.numpy()
                
                out = self.model(x)

                total_loss, losses = self.model.loss(out, y) 
                for k, v in losses.items():
                    loss[k] = loss.get(k, []) + [v]
                for k in y.keys():  # remove -1 On training set
                    mask = y[k] != -1
                    y[k] = y[k][mask]
                    out[k] = out[k][mask]
                for k, v in out.items():
                    outs[k] = outs.get(k, []) + v.detach().cpu().numpy().tolist()
                for k, v in y.items():
                    true[k] = true.get(k, []) + v.detach().cpu().numpy().tolist()

                total_loss /= self.acc_step
                total_loss.backward()
                if i % self.acc_step == 0 or i == len(train_loader):  # i starts from 1
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                pbar.update(1)
            
            loss_dict, metrics_dict = {}, {}
            for k, v in loss.items():
                loss_dict[f"loss_{k}_train_sample"] = np.mean(v)
            all_metrics = self.model.metrics(outs, true)
            for k, metrics in all_metrics.items():
                for m, a in metrics.items():
                    if m != 'confusion_matrix':
                        metrics_dict[f"{m}_{k}_train_sample"] = a
            wandb.log(loss_dict, step=self.epoch)
            wandb.log(metrics_dict, step=self.epoch)
            
            self.loss_history.append(float(loss_dict['loss_total_train_sample']))
            self.log.write(f"train_with_sampler epoch_{self.epoch} : {loss_dict}")
            self.log.write(f'metrics : ' + str(metrics_dict))
            pbar.close()
    
    def eval_epoch(self, val_loader, mode='valid'):
        self.model.eval()
        with torch.no_grad():
            loss = {}
            outs = {}
            true = {}
            for i, (x, y, p_idxs) in enumerate(val_loader, 1):
                x = x.cuda()
                for k, v in y.items():
                    y[k] = v.cuda()
                p_idxs = p_idxs.cpu().data.numpy()
                
                out = self.model(x)
                
                _, losses = self.model.loss(out, y) 
                for k, v in losses.items():
                    loss[k] = loss.get(k, []) + [v]
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
                    if m != 'confusion_matrix':
                        metrics_dict[f"{m}_{k}_{mode}"] = a
            wandb.log(loss_dict, step=self.epoch)
            wandb.log(metrics_dict, step=self.epoch)
            
            self.log.write(f"{mode} epoch_{self.epoch} : {loss_dict}")
            self.log.write(f'metrics : ' + str(metrics_dict))

        # save best model of valid
        if mode in ['valid']:
            tasks = eval(self.args.use_tasks)
            for task in tasks:
                score = get_score(metrics_dict, task)
                if score > self.best_score:
                    self.best_score = score
                    self.best_metrics = metrics_dict
                    save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'{mode}_{task}_Best.pth'))
    
    def on_epoch_end(self):
        save_trainer(self, os.path.join(self.ckpt_path, 'Final_Trainer.pkl'))
        save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'Final.pth'))
        self.epoch += 1  
    
        