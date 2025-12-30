from typing import Any
from settings.loss import FocalLoss
from torch.nn import CrossEntropyLoss
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import torch
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

# MultiTaskMI argutments:
# loss_weight : str
# use_tasks : str 


def GetHead(in_dim, out_dim):
    return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, out_dim)
        )
    

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, in_feat, args) -> None:
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        # support tasks
        self.support_tasks = \
            ['TD', 'CE', 'TI', 'REC', 'PI', 'LNM']
        # tasks
        tasks = {
            'TD': TumorDiffTask(in_feat),          
            'REC': RecTask(in_feat),              
            'CE': CancerEmbolusTask(in_feat),     
            'TI': InvasionTask(in_feat),          
            'PI': NerveInvasionTask(in_feat),    
            'LNM': LNMTask(in_feat)
        }
        use_tasks = set(eval(args.use_tasks))
        print("Use tasks : ", use_tasks, flush=True)
        for task_name in self.support_tasks:
            if task_name not in use_tasks:
                tasks.pop(task_name)
        self.tasks = nn.ModuleDict(tasks)
        
        print("Params Size:", sum(p.numel() for p in self.parameters()), flush=True)
    
    def get_backbone_params(self):
        return self.backbone.get_backbone_params()
    
    def get_others_params(self):
        backbones = set(self.get_backbone_params())
        return [p for p in self.parameters() if p not in backbones]
        
    def loss(self, outputs, targets) -> Any:
        total_loss = 0
        loss = {}
        for task_name, task in self.tasks.items():
            out = outputs[task_name]
            tar = targets[task_name]
            task_loss = task.loss(out, tar)
            loss[task_name] = task_loss
            total_loss += task_loss
        loss['total'] = total_loss
        return total_loss, loss

    def metrics(self, outputs, targets) -> Any:
        metrics = {}
        for task_name, task in self.tasks.items():
            out = outputs[task_name]
            tar = targets[task_name]
            metrics[task_name] = task.metrics(torch.tensor(out), np.array(tar))
        return metrics
    
    def forward(self, *args, **kargs):
        feat_p = self.backbone(*args, **kargs)
        outs = {}
        for task_name, task in self.tasks.items():
            outs[task_name] = task(feat_p)
        return outs


class TumorDiffTask(nn.Module):
    def __init__(self, in_feat) -> None:
        super(TumorDiffTask, self).__init__()
        self.head = GetHead(in_feat, 3)
    
    def loss(self, out, target):
        mask = target != -1  # 去除tasget中的-1的sample
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, requires_grad=True).float().to(out.device)
        out = out[mask]
        target = target[mask]
        return F.cross_entropy(out, target)

    def metrics(self, out, target):
        probs = nn.Softmax(dim=1)(out).detach().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int32)
        target_onehot = label_binarize(target, classes=[0, 1, 2])
        try:
            auc = roc_auc_score(target_onehot, probs, average='macro', multi_class='ovr')
        except Exception as e:
            auc = 0
            print(f"Error on calculate {self.__class__.__name__} AUC: ", e)
        return {
            'Acc': accuracy_score(target, pred),
            'AUC': auc,
            'F1': f1_score(target, pred, average='macro'),
            'Precision': precision_score(target, pred, average='macro'),
            'Recall': recall_score(target, pred, average='macro'),
            'confusion_matrix': confusion_matrix(target, pred)
        }
    
    def forward(self, out):
        return self.head(out)


class RecTask(nn.Module):
    def __init__(self, in_feat) -> None:
        super(RecTask, self).__init__()
        self.head = GetHead(in_feat, 2)
    
    def loss(self, out, target):
        mask = target != -1  # 去除tasget中的-1的sample
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, requires_grad=True).float().to(out.device)
        out = out[mask]
        target = target[mask]
        return FocalLoss()(out, target)
    
    def metrics(self, out, target):
        probs = torch.softmax(out, dim=1).detach().numpy()
        probs = probs[:, 1]  # 只取正类概率 
        pred = [1 if p >= 0.5 else 0 for p in probs]
        return {
            'Acc': accuracy_score(target, pred),
            'AUC': roc_auc_score(target, probs),  # auc default average='macro'
            'F1': f1_score(target, pred, average='macro'),
            'Precision': precision_score(target, pred, average='macro'),
            'Recall': recall_score(target, pred, average='macro'),
            'confusion_matrix': confusion_matrix(target, pred)
        }
    
    def forward(self, out):
        return self.head(out)
        
    
class CancerEmbolusTask(nn.Module):
    def __init__(self, in_feat) -> None:
        super(CancerEmbolusTask, self).__init__()
        self.head = GetHead(in_feat, 2)
    
    def loss(self, out, target):
        mask = target != -1  # 去除tasget中的-1的sample
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, requires_grad=True).float().to(out.device)
        out = out[mask]
        target = target[mask]
        return FocalLoss()(out, target)
    
    def metrics(self, out, target):
        probs = nn.Softmax(dim=1)(out).detach().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int32)
        conf = confusion_matrix(target, pred)
        probs = probs[:, 1]  # 只取正类概率 
        return {
            'Acc': accuracy_score(target, pred),
            'AUC': roc_auc_score(target, probs),
            'F1': f1_score(target, pred, average='macro'),
            'Precision': precision_score(target, pred, average='macro'),
            'Recall': recall_score(target, pred, average='macro'),
            'confusion_matrix': conf 
        }
        
    def forward(self, out):
        return self.head(out)


class InvasionTask(nn.Module):
    def __init__(self, in_feat) -> None:
        super(InvasionTask, self).__init__()
        self.head = GetHead(in_feat, 2)
    
    def loss(self, out, target):
        mask = target != -1  # 去除tasget中的-1的sample
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, requires_grad=True).float().to(out.device)
        out = out[mask]
        target = target[mask]
        return FocalLoss()(out, target)
    
    def metrics(self, out, target):
        probs = nn.Softmax(dim=1)(out).detach().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int32)
        probs = probs[:, 1]  # 只取正类概率 
        conf = confusion_matrix(target, pred)
        return {
            'Acc': accuracy_score(target, pred),
            'AUC': roc_auc_score(target, probs),
            'F1': f1_score(target, pred, average='macro'),
            'Precision': precision_score(target, pred, average='macro'),
            'Recall': recall_score(target, pred, average='macro'),
            'confusion_matrix': conf 
        }
        
    def forward(self, out):
        return self.head(out)
    

class NerveInvasionTask(nn.Module):
    def __init__(self, in_feat) -> None:
        super(NerveInvasionTask, self).__init__()
        self.head = GetHead(in_feat, 2)
    
    def loss(self, out, target):
        mask = target != -1  # 去除tasget中的-1的sample
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, requires_grad=True).float().to(out.device)
        out = out[mask]
        target = target[mask]
        return FocalLoss()(out, target)
    
    def metrics(self, out, target):
        probs = nn.Softmax(dim=1)(out).detach().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int32)
        probs = probs[:, 1]  # 只取正类概率 
        conf = confusion_matrix(target, pred)
        return {
            'Acc': accuracy_score(target, pred),
            'AUC': roc_auc_score(target, probs),
            'F1': f1_score(target, pred, average='macro'),
            'Precision': precision_score(target, pred, average='macro'),
            'Recall': recall_score(target, pred, average='macro'),
            'confusion_matrix': conf 
        }
        
    def forward(self, out):
        return self.head(out)
    

class LNMTask(nn.Module):
    def __init__(self, in_feat) -> None:
        super(LNMTask, self).__init__()
        self.head = GetHead(in_feat, 2)
    
    def loss(self, out, target):
        mask = target != -1  # 去除tasget中的-1的sample
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, requires_grad=True).float().to(out.device)
        out = out[mask]
        target = target[mask]
        return FocalLoss()(out, target)
    
    def metrics(self, out, target):
        probs = nn.Softmax(dim=1)(out).detach().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int32)
        probs = probs[:, 1]  # 只取正类概率 
        conf = confusion_matrix(target, pred)
        return {
            'Acc': accuracy_score(target, pred),
            'AUC': roc_auc_score(target, probs),
            'F1': f1_score(target, pred, average='macro'),
            'Precision': precision_score(target, pred, average='macro'),
            'Recall': recall_score(target, pred, average='macro'),
            'confusion_matrix': conf 
        }
        
    def forward(self, out):
        return self.head(out)