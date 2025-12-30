import matplotlib.pyplot as plt
import os
from modules.min_norm_solvers import MinNormSolver, gradient_normalizers
import tqdm
import numpy as np
import torch
from torch.cuda.amp import autocast
from modules.trainer import Trainer

        
class ParetoTrainer(Trainer):
    def __init__(self, fold=0, args=None):  
        super().__init__(fold, args)
        
        self.normalization_type = "loss+"
        self.grad_backup = {}  # gradients backup for accumulating
        self.tasks = list(self.model.module.tasks.keys()) if self.args.use_ddp else list(self.model.tasks.keys())

        self.recalculate_weights_step = 5  # recalculate weights every 5 epochs
        self.average_scale_list = []
        self.average_scale = None
        self.scale_list = [[] for _ in range(len(self.tasks))]

    def calc_pareto_weights(self, x, y):
        # forward once to get the loss
        # backward six times to get the gradients
        loss_data = {}
        scale = {}
        grads = {}
        
        num_of_task = len(self.tasks)
        out = self.model(x)
        _, tasks_loss = self.LossFn(out, y)
       
        for t in self.tasks:
            self.model.zero_grad()
            loss_data[t] = tasks_loss[t].item()
            tasks_loss[t].backward(retain_graph=True)  # can't use ddp in pareto because of redudant grads
            
            grads[t] = []
            key_params = self.model.module.backbone if self.args.use_ddp else self.model.backbone
            for name, param in key_params.named_parameters():
                if param.grad is not None:
                    grads[t].append(torch.autograd.Variable(param.grad.data.clone(), requires_grad=False))

        # gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu().numpy() for gr in grads[t]]))
        # that is the loss * the l2 norm of the gradients
        gn = gradient_normalizers(grads, loss_data, self.normalization_type)
        for t in self.tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]  # normalize to get the gradient direction

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in self.tasks])
        for i, t in enumerate(self.model.tasks.keys()):
            scale[t] = float(sol[i])
        
        scale_tensor = torch.tensor([scale[t] for t in self.model.tasks.keys()]).to(self.device)
        return scale_tensor * num_of_task  # (1, 6)

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
                
                if self.epoch % self.recalculate_weights_step == 0 or self.average_scale is None:
                    scale = self.calc_pareto_weights(x, y).to(self.device)
                    self.average_scale_list.append(scale)  # record the scale for each batch
                    self.average_scale = None
                    if self.local_rank == 0:
                        for j, _ in enumerate(self.tasks):
                            self.scale_list[j].append(scale[j].item())
                else:
                    scale = self.average_scale.clone()
                
                self.model.zero_grad()
                if self.args.use_amp:
                    with autocast():
                        out = self.model(x)
                        _, losses = self.LossFn(out, y)
                        tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                        weighted_loss = torch.sum(scale * tasks_loss)
                        weighted_loss = self.scaler.scale(weighted_loss)
                        weighted_losses = {k: scale[i] * losses[k] for i, k in enumerate(losses.keys())}
                else:
                    out = self.model(x)
                    _, losses = self.LossFn(out, y)
                    tasks_loss = torch.stack([losses[k] for k in losses.keys()])    # shape = (n_tasks, )
                    weighted_loss = torch.sum(scale * tasks_loss)
                    weighted_losses = {k: scale[i] * losses[k] for i, k in enumerate(losses.keys())}
                    
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
                
                del x, y, weighted_loss, losses
                pbar.update(1)
            pbar.close()
            self.on_loader_exit('train_sample', loss, outs, true)
            
            # draw the scale when epoch % recalculate_weights_step == 0
            if (self.epoch % self.recalculate_weights_step == 0 or self.average_scale is None):
                if self.local_rank == 0:
                    plt.cla()
                    fig, ax = plt.subplots(2, 3)
                    ax = ax.flatten()
                    X = np.arange(len(self.scale_list[0]))
                    for i, w_list in enumerate(self.scale_list):
                        ax[i].plot(X, w_list, label=f'task_{i}',linewidth = 0.5)
                        ax[i].plot(X, [1] * len(X), 'r--')  # no label
                        ax[i].legend()
                    plt.savefig(os.path.join(self.log_path, f'loss_weight.png'))
                # average scale
                self.average_scale = torch.mean(torch.stack(self.average_scale_list), dim=0).to(self.device)
                self.average_scale_list = []
                    
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
                
                if (self.epoch > 0) and (self.average_scale is not None):
                    scale = self.average_scale.clone()
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
                
                del x, y, out, weighted_loss, losses
            self.on_loader_exit(mode, loss, outs, true)