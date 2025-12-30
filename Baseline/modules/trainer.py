from datasets import GetDataLoader
from models import GetModel
from settings import GetOptimizer, GetScheduler
from utils import Logger, save_trainer, save_model, load_model, load_trainer
import os
import tqdm
import numpy as np
import wandb
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

def get_score(metrics, key_task):
    score = 0.6 * metrics[f"AUC_{key_task}_valid"] + 0.1 * metrics[f"F1_{key_task}_valid"] + 0.3 * metrics[f"Acc_{key_task}_valid"]
    return score

        
class Trainer:
    def __init__(self, fold=0, args=None):  
        assert args is not None, 'Please input args!!!'
        self.args = args
        self.fold = fold
        
        if self.args.use_ddp:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ['LOCAL_RANK'])
            print("Rank : ", self.local_rank, flush=True)
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.model = GetModel(self.args).cuda()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                            self.model, device_ids=[self.local_rank], 
                            output_device=self.local_rank, static_graph=True)
            self.optimizer = GetOptimizer(self.args, self.model.module)
            self.scheduler  = GetScheduler(self.args, self.optimizer)
        else:
            self.device = torch.device("cuda")
            self.local_rank = 0
            self.model = GetModel(self.args).cuda()
            self.optimizer = GetOptimizer(self.args, self.model)
            self.scheduler  = GetScheduler(self.args, self.optimizer)
            
        # dataset 
        mead_std = ([175.14728804175988, 110.57123792228117, 176.73598615775617], [21.239463551725915, 39.15991384752335, 10.99100631656543])
        self.train_loader, self.val_loader, self.test_loader = \
            GetDataLoader(fold, mead_std, self.args, False)
        print("MeanStd = " + str(self.train_loader.dataset.mean_std)) 
        
        self.epoch = 0
        self.iters = 0
        self.acc_step = self.args.acc_step  # accumulate_step
            
        if self.args.continue_training:
            ckp_trainer = load_trainer(self.args.load_pth_path)
            self.epoch = ckp_trainer.epoch
            print(f"Continue training from {self.args.load_pth_path} !!!", flush=True)
            print("Now epoch : ", self.epoch, flush=True)
            self.optimizer.load_state_dict(ckp_trainer.optimizer.state_dict())
            self.scheduler.load_state_dict(ckp_trainer.scheduler.state_dict())
            self.model.load_state_dict(ckp_trainer.model.state_dict())
            del ckp_trainer
        elif self.args.finetune:
            print(f"Fine-tune from {self.args.load_pth_path}!!!", flush=True)
            cp = load_model(self.args.load_pth_path)
            pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
            pretrain = {k: v for k, v in pretrain.items() if k in self.model.state_dict()}
            self.model.load_state_dict(pretrain)
            
        # early stop
        self.loss_history = []
        self.patience = 150
        self.monitor_length = 20  # monitor the last 20 epochs
        
        # amp
        if self.args.use_amp:
            self.scaler = GradScaler()
            
        # multi-task or single-task
        if len(eval(self.args.use_tasks)) > 1:
            self.multi_task_best_score = 0
            self.best_multi_task_metrics = {}
        else:
            self.best_metrics = {}
            self.best_score = 0
        
        # trainer config
        run_path = [self.args.model, self.args.runs_id, 'fold'+str(fold)][:2+int(fold>0)]
        self.log_path = os.path.join(self.args.log_path, *run_path)
        self.ckpt_path = os.path.join(self.args.ckpt_path, *run_path) 
        print("log_path : ", self.log_path, flush=True)
        print("ckpt_path : ", self.ckpt_path, flush=True)
        
        if os.path.exists(os.path.join(self.ckpt_path, 'Final_Trainer.pkl')):
            print("Trainer already exists!!!", flush=True)
            raise ValueError("Trainer already exists!!!")
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.log = Logger(os.path.join(self.log_path, 'log.txt'))
        self.log.write("settings : " + str(args))
        self.log.write("MeanStd = " + str(self.train_loader.dataset.mean_std))
    
    def early_stop(self):
        # early stopping
        epoch_run = len(self.loss_history)
        early_stop_flag= torch.zeros(1).to(self.device)
        if self.local_rank == 0 and epoch_run > self.patience and epoch_run > self.monitor_length:
            if self.loss_history[-1] >= np.mean(self.loss_history[-self.monitor_length:]):
                early_stop_flag += 1  # kill all processes at this time
        if self.args.use_ddp:      
            dist.all_reduce(early_stop_flag,op=dist.ReduceOp.SUM)
        if early_stop_flag.item() == 1:
            print("Early stopping!!! on  epoch " + str(self.epoch), flush=True)
            self.log.write("Early stopping!!! on epoch " + str(self.epoch))
            return True
        return False
    
    def run(self):
        start_epoch = self.epoch
        for epoch_id in range(start_epoch, self.args.num_epochs + 1):  
            # training
            if epoch_id > start_epoch:  # start_epoch 不训练
                self.train_epoch(self.train_loader)
                self.scheduler.step()
                if self.local_rank == 0:
                    for idx, groups in enumerate(self.optimizer.param_groups):
                        wandb.log({f"lr_{['backbones', 'head'][idx]}": groups['lr']}, step=self.epoch)
            # evaluation
            if self.local_rank == 0:
                if self.val_loader is not None:
                    self.eval_epoch(self.val_loader, 'valid')
                if self.test_loader is not None:
                    self.eval_epoch(self.test_loader, 'test')
                self.on_epoch_end()
            # early stopping
            if self.early_stop():
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
                
                if self.args.use_amp:
                    with autocast():
                        out = self.model(x)
                        total_loss, losses = self.LossFn(out, y)
                        total_loss = self.scaler.scale(total_loss)
                else:
                    out = self.model(x)
                    total_loss, losses = self.LossFn(out, y)
                
                loss['total'] = loss.get('total', []) + [total_loss.item()]
                for k, v in losses.items():
                    loss[k] = loss.get(k, []) + [v.item()]
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
                    if self.args.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                del x, y, out, total_loss, losses
                pbar.update(1)
            pbar.close()
            self.on_loader_exit('train_sample', loss, outs, true)
    
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
                total_loss, losses = self.LossFn(out, y)
                
                loss['total'] = loss.get('total', []) + [total_loss.item()]
                for k, v in losses.items():
                    loss[k] = loss.get(k, []) + [v.item()]
                for k, v in out.items():
                    outs[k] = outs.get(k, []) + v.detach().cpu().numpy().tolist()
                for k, v in y.items():
                    true[k] = true.get(k, []) + v.detach().cpu().numpy().tolist()
                    
        self.on_loader_exit(mode, loss, outs, true) 
            
    def on_epoch_end(self):
        save_trainer(self, os.path.join(self.ckpt_path, 'Final_Trainer.pkl'))
        save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'Final.pth'))
        torch.cuda.empty_cache()
        if len(eval(self.args.use_tasks)) > 1:
            self.log.write(f"Multi-Task Best Score : {self.multi_task_best_score}")
            self.log.write(f"Multi-Task Best Metrics : {self.best_multi_task_metrics}")
        else:
            self.log.write(f"Best Score : {self.best_score}")
            self.log.write(f"Best Metrics : {self.best_metrics}")
        self.epoch += 1  
    
    def save_best_model(self, metrics_dict):
        if self.local_rank == 0:
            tasks = eval(self.args.use_tasks)
            multi_task_score = 0
            for task in tasks:
                score = get_score(metrics_dict, task)
                multi_task_score += score
                if len(tasks) == 1 and score > self.best_score:
                    self.best_score = score
                    self.best_metrics = metrics_dict
                    save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'valid_{task}_Best.pth'))
            if len(tasks) > 1 and multi_task_score > self.multi_task_best_score:
                # if multi-task, save the total_best_score model
                self.multi_task_best_score = multi_task_score
                self.best_multi_task_metrics = metrics_dict
                save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'valid_MultiTask_Best.pth'))       
            
    def on_loader_exit(self, mode, loss, outs, true):
        loss_dict, metrics_dict = {}, {}
        for k, v in loss.items():
            loss_dict[f"loss_{k}_{mode}"] = np.mean(v)
        all_metrics = self.Metrics(outs, true)
        for k, metrics in all_metrics.items():
            for m, a in metrics.items():
                if m != 'confusion_matrix':
                    metrics_dict[f"{m}_{k}_{mode}"] = a
        
        if self.local_rank == 0:
            if mode == 'train':
                if f'loss_total_{mode}' in loss_dict:
                    self.loss_history.append(float(loss_dict[f'loss_total_{mode}']))
                elif f'loss_weighted_total_{mode}' in loss_dict:
                    self.loss_history.append(float(loss_dict[f'loss_weighted_total_{mode}']))
                else:
                    raise ValueError("No loss_total or loss_weighted_total !!!")
            # log to wandb and log.txt
            wandb.log(loss_dict, step=self.epoch)
            wandb.log(metrics_dict, step=self.epoch)
            self.log.write(f"{mode} epoch_{self.epoch} : {loss_dict}")
            self.log.write(f'metrics : ' + str(metrics_dict))
            
        # save best model of valid
        if mode in ['valid']:
            self.save_best_model(metrics_dict)   
            
    def LossFn(self, x, y):
        return self.model.module.loss(x, y) if self.args.use_ddp else self.model.loss(x, y)
    
    def Metrics(self, x, y):
        return self.model.module.metrics(x, y) if self.args.use_ddp else self.model.metrics(x, y)