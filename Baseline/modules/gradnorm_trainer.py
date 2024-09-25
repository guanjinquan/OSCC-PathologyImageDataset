from utils import save_model
import os
import tqdm
import numpy as np
import wandb
import torch
from torch.cuda.amp import autocast
from modules.trainer import Trainer
from utils import load_model

        
class GradNormTrainer(Trainer):
    def __init__(self, fold=0, args=None):  
        super().__init__(fold, args)
        
        # loss weights for each task [register as a parameter]
        self.model.loss_weights = torch.nn.Parameter(torch.ones(len(eval(self.args.use_tasks))).cuda())
        self.initial_task_loss = None
        
        # GradNorm hyperparameter
        self.alpha = 0.12
        self.loss_weights_grad = []
        
        # load loss_weight pretrain
        if self.args.finetune:
            assert self.args.load_pth_path is not None, "load_path can't be None."
            print(f"Load from {self.args.load_pth_path}!!!", flush=True)
            cp = load_model(self.args.load_pth_path)
            pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
            pretrain = {k: v for k, v in pretrain.items() if k in self.model.state_dict()}
            self.model.loss_weights = torch.nn.Parameter(pretrain['loss_weights'].cuda())
        
    def grad_norm_method(self, task_loss):
        # get layer of shared weights
        W = self.model.fusion_block  # neck is the last shared layer in our model

        if self.initial_task_loss is None:
            # set L(0)
            if torch.cuda.is_available():
                self.initial_task_loss = task_loss.data.cpu()
            else:
                self.initial_task_loss = task_loss.data
            self.initial_task_loss = self.initial_task_loss.numpy()
        
        # get the gradient norms for each of the tasks G^{(i)}_w(t) 
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(self.model.loss_weights[i], gygw[0])))
        norms = torch.stack(norms)

        # compute the inverse training rate r_i(t)  \curl{L}_i 
        if torch.cuda.is_available():
            loss_ratio = task_loss.data.cpu().numpy() / self.initial_task_loss
        else:
            loss_ratio = task_loss.data.numpy() / self.initial_task_loss
        # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # compute the mean norm \tilde{G}_w(t) 
        if torch.cuda.is_available():
            mean_norm = np.mean(norms.data.cpu().numpy())
        else:
            mean_norm = np.mean(norms.data.numpy())

        # compute the GradNorm loss 
        # this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
            
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

        # compute the gradient for the weights
        return torch.autograd.grad(grad_norm_loss, self.model.loss_weights)[0]
        
    def train_epoch(self, train_loader):
        self.model.train()
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            loss = {}
            outs = {}
            true = {}
            self.optimizer.zero_grad()            
            for i, (x, y, p_idxs) in enumerate(train_loader, 1):
                
                if self.local_rank == 0:
                    # log the loss weights
                    for j, task in enumerate(eval(self.args.use_tasks)):
                        wandb.log({f'loss_weight_{task}': self.model.loss_weights[j].item()})
                
                x = x.cuda()
                for k, v in y.items():
                    y[k] = v.cuda()
                p_idxs = p_idxs.cpu().data.numpy()
                
                if self.args.use_amp:
                    with autocast():
                        out = self.model(x)
                        _, losses = self.LossFn(out, y)
                        tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                        weighted_loss = torch.sum(self.model.loss_weights * tasks_loss)
                        weighted_loss = self.scaler.scale(weighted_loss)
                else:
                    out = self.model(x)
                    _, losses = self.LossFn(out, y)
                    tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                    weighted_loss = torch.sum(self.model.loss_weights * tasks_loss)
                    
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
                weighted_loss.backward(retain_graph=True)
                
                # set the gradients of Lw_i(t) to zero
                self.model.loss_weights.grad.data = self.model.loss_weights.grad.data * 0.0
                self.loss_weights_grad.append(self.grad_norm_method(tasks_loss).clone())
                if i % self.acc_step == 0 or i == len(train_loader):  # i starts from 1
                    self.model.loss_weights.grad = torch.mean(torch.stack(self.loss_weights_grad), dim=0)
                    print(f"loss_weights_grad: {self.model.loss_weights.grad}", flush=True)
                    self.log.write(f"loss_weights_grad: {self.model.loss_weights.grad}\n")
                    
                    exit(0)
                    
                    self.loss_weights_grad.clear()
                    if self.args.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                del x, y, out, weighted_loss, losses
                pbar.update(1)
            pbar.close()
            self.on_loader_exit('train', loss, outs, true)
                            
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
                _, losses = self.LossFn(out, y)
                tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                weighted_loss = torch.sum(self.model.loss_weights * tasks_loss)
                
                loss['weighted_total'] = loss.get('weighted_total', []) + [weighted_loss.item()]
                for k, v in losses.items():
                    loss[k] = loss.get(k, []) + [v.item()]
                for k, v in out.items():
                    outs[k] = outs.get(k, []) + v.detach().cpu().numpy().tolist()
                for k, v in y.items():
                    true[k] = true.get(k, []) + v.detach().cpu().numpy().tolist()
                
        self.on_loader_exit(mode, loss, outs, true)