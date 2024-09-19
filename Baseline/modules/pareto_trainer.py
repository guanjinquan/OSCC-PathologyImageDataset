import sys
import os
from modules.min_norm_solvers import MinNormSolver, gradient_normalizers
from utils import save_model
import tqdm
import numpy as np
import wandb
import torch
from torch.cuda.amp import autocast
from modules.trainer import Trainer, get_score

        
class ParetoTrainer(Trainer):
    def __init__(self, fold=0, args=None):  
        super().__init__(fold, args)
        
        self.normalization_type = "loss+"
        self.grad_backup = {}  # gradients backup for accumulating
        self.validation_scale = torch.tensor([1.0 for _ in self.model.tasks.keys()]).to(self.device)

    def calc_pareto_weights(self, x, y):
        # forward once to get the loss
        # backward six times to get the gradients
        
        loss_data = {}
        scale = {}
        grads = {}
        tasks = list(self.model.tasks.keys())
        
        out = self.model(x)
        _, tasks_loss = self.model.loss(out, y)
       
        for t in tasks:
            print("Working on task: ", t, flush=True)
            self.model.zero_grad()
            loss_data[t] = tasks_loss[t].item()
            tasks_loss[t].backward(retain_graph=True)
            
            grads[t] = []
            for name, param in self.model.backbone.named_parameters():
                if param.grad is not None:
                    grads[t].append(torch.autograd.Variable(param.grad.data.clone(), requires_grad=False))

        gn = gradient_normalizers(grads, loss_data, self.normalization_type)
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t] # type: tensor

        print("Calculating scales", flush=True)
        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
        for i, t in enumerate(self.model.tasks.keys()):
            scale[t] = float(sol[i])
        
        scale_tensor = torch.tensor([scale[t] for t in self.model.tasks.keys()]).to(self.device)
        return scale_tensor

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.grad_backup:
                    self.grad_backup[name] = param.grad.clone()
                else:
                    self.grad_backup[name] += param.grad.clone()
                    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

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
                
                scale = self.calc_pareto_weights(x, y).to(self.device)
                self.validation_scale += scale
                
                self.model.zero_grad()
                if self.args.use_amp:
                    with autocast():
                        out = self.model(x)
                        _, losses = self.model.loss(out, y)
                        tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                        weighted_loss = torch.sum(scale * tasks_loss)
                        weighted_loss = self.scaler.scale(weighted_loss)
                else:
                    out = self.model(x)
                    _, losses = self.model.loss(out, y) 
                    tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                    weighted_loss = torch.sum(scale * tasks_loss)
                    
                loss['weighted_total'] = loss.get('weighted_total', []) + [weighted_loss.item()]
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

                weighted_loss /= self.acc_step
                weighted_loss.backward()
                self.backup_grad()
                # set the gradients of Lw_i(t) to zero
                if i % self.acc_step == 0 or i == len(train_loader):  # i starts from 1
                    self.restore_grad()
                    if self.args.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                del x, y, out, weighted_loss, losses
                
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
            
            self.loss_history.append(float(loss_dict['loss_weighted_total_train_sample']))
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
                tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                
                if self.epoch > 0:
                    scale = self.validation_scale / self.epoch
                    weighted_loss = torch.sum(scale * tasks_loss)
                else:
                    weighted_loss = torch.sum(tasks_loss)
                
                loss['weighted_total'] = loss.get('weighted_total', []) + [weighted_loss.item()]
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
                    if m != 'confusion_matrix':
                        metrics_dict[f"{m}_{k}_{mode}"] = a
            wandb.log(loss_dict, step=self.epoch)
            wandb.log(metrics_dict, step=self.epoch)
            
            self.log.write(f"{mode} epoch_{self.epoch} : {loss_dict}")
            self.log.write(f'metrics : ' + str(metrics_dict))

        if mode in ['valid']:
            tasks = eval(self.args.use_tasks)
            multi_task_score = 0
            for task in tasks:
                score = get_score(metrics_dict, task)
                multi_task_score += score
            if len(tasks) > 1 and multi_task_score > self.multi_task_best_score:
                # if multi-task, save the total_best_score model
                self.multi_task_best_score = multi_task_score
                self.best_multi_task_metrics = metrics_dict
                save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'{mode}_MultiTask_Best.pth'))     
            