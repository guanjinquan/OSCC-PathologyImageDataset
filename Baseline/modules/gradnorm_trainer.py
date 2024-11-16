import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from modules.trainer import Trainer
from utils import load_model

        
class GradNormTrainer(Trainer):
    def __init__(self, fold=0, args=None):  
        super().__init__(fold, args)
        
        # loss weights for each task [register as a parameter]
        self.loss_weights = torch.nn.Parameter(torch.ones(len(eval(self.args.use_tasks))).cuda())
        self.GradNormOptimizer = torch.optim.Adam([self.loss_weights], lr=1e-5)
        self.initial_task_loss = None
        
        # GradNorm hyperparameter
        self.alpha = 0.12
        self.loss_weights_grad = []
        
        # load loss_weight pretrain [didn't save loss_weights in the best checkpoint]
        # if self.args.finetune:
        #     assert self.args.load_pth_path is not None, "load_path can't be None."
        #     print(f"Load from {self.args.load_pth_path}!!!", flush=True)
        #     cp = load_model(self.args.load_pth_path)
        #     pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
        #     pretrain = {k: v for k, v in pretrain.items() if k in self.model.state_dict()}
        #     self.loss_weights = torch.nn.Parameter(pretrain['loss_weights'].cuda())
        
        self.loss_weight_list = [[] for _ in range(len(eval(self.args.use_tasks)))]
        self.loss_weight_grad_list = [[] for _ in range(len(eval(self.args.use_tasks)))]
        
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
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)  # 计算共享层的梯度
            # compute the norm
            norms.append(torch.norm(torch.mul(self.loss_weights[i], gygw[0])))  # 计算共享层的梯度范数[乘上权重之后的]
        norms = torch.stack(norms)

        # compute the inverse training rate r_i(t)  \curl{L}_i 
        if torch.cuda.is_available():
            loss_ratio = task_loss.data.cpu().numpy() / self.initial_task_loss
        else:
            loss_ratio = task_loss.data.numpy() / self.initial_task_loss
        # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)  # 如果相对于初始loss下降的比例大于平均值，则r_i(t) > 1，此时应该减小该任务的权重

        # compute the mean norm \tilde{G}_w(t) 
        if torch.cuda.is_available():
            mean_norm = np.mean(norms.data.cpu().numpy())  # 计算所有任务的梯度范数的均值
        else:
            mean_norm = np.mean(norms.data.numpy())

        # compute the GradNorm loss 
        # this term has to remain constant
        # inverse_train_rate < 1时，constant_term > mean_norm，此时应该增大该任务的权重
        # inverse_train_rate > 1时，constant_term < mean_norm，此时应该减小该任务的权重
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)  
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
            
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))  # 范数与常数项的差值的绝对值之和，目的是让所有任务的梯度范数接近常数项

        # compute the gradient for the weights
        return torch.autograd.grad(grad_norm_loss, self.loss_weights)[0]
        
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
                        _, losses = self.LossFn(out, y)
                        tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                        weighted_loss = torch.sum(self.loss_weights * tasks_loss)
                        weighted_loss = self.scaler.scale(weighted_loss)
                        weighted_losses = {k: self.loss_weights[i] * losses[k] for i, k in enumerate(losses.keys())}
                else:
                    out = self.model(x)
                    _, losses = self.LossFn(out, y)
                    tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                    weighted_loss = torch.sum(self.loss_weights * tasks_loss)
                    weighted_losses = {k: self.loss_weights[i] * losses[k] for i, k in enumerate(losses.keys())}
                           
                loss['weighted_total'] = loss.get('weighted_total', []) + [weighted_loss.item()]
                for k, v in weighted_losses.items():
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
                self.loss_weights.grad.data = self.loss_weights.grad.data * 0.0
                self.loss_weights_grad.append(self.grad_norm_method(tasks_loss).clone())
                if i % self.acc_step == 0 or i == len(train_loader):  # i starts from 1
                    self.loss_weights.grad = torch.mean(torch.stack(self.loss_weights_grad), dim=0)
                    
                    # loss weights record
                    for i, w in enumerate(self.loss_weights):
                        self.loss_weight_list[i].append(w.item())
                    for i, w in enumerate(self.loss_weights.grad):
                        self.loss_weight_grad_list[i].append(w.item())
                    
                    # Loss weights update
                    self.GradNormOptimizer.step()
                    self.GradNormOptimizer.zero_grad()
                    self.loss_weights_grad.clear()
                    
                    # model weights update
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

            self.on_loader_exit('train_sample', loss, outs, true)
            
            # draw loss weights curve
            if self.local_rank == 0:
                plt.cla()
                fig, ax = plt.subplots(2, 3)
                ax = ax.flatten()
                X = np.arange(len(self.loss_weight_list[0]))
                for i, w_list in enumerate(self.loss_weight_list):
                    ax[i].plot(X, w_list, label=f'task_{i}',linewidth = 0.5)
                    ax[i].plot(X, [1] * len(X), 'r--')  # no label
                    ax[i].legend()
                plt.savefig(os.path.join(self.log_path, f'loss_weight.png'))
                plt.cla()
                fig, ax = plt.subplots(2, 3)
                ax = ax.flatten()
                X = np.arange(len(self.loss_weight_grad_list[0]))
                for i, w_list in enumerate(self.loss_weight_grad_list):
                    ax[i].plot(X, w_list, label=f'task_{i}',linewidth = 0.5)
                    ax[i].plot(X, [1] * len(X), 'r--')  # no label
                    ax[i].legend()
                plt.savefig(os.path.join(self.log_path, f'loss_weight_grad.png'))
                            
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
                weighted_loss = torch.sum(self.loss_weights * tasks_loss)
                weighted_losses = {k: self.loss_weights[i] * losses[k] for i, k in enumerate(losses.keys())}
                
                loss['weighted_total'] = loss.get('weighted_total', []) + [weighted_loss.item()]
                for k, v in weighted_losses.items():
                    loss[k] = loss.get(k, []) + [v.item()]
                for k, v in out.items():
                    outs[k] = outs.get(k, []) + v.detach().cpu().numpy().tolist()
                for k, v in y.items():
                    true[k] = true.get(k, []) + v.detach().cpu().numpy().tolist()
                
        self.on_loader_exit(mode, loss, outs, true)