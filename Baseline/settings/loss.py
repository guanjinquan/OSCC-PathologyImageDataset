import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BalancedLoss(nn.Module):
    def __init__(self, num, reduction='mean'):
        super(BalancedLoss, self).__init__()
        self.reduction = reduction
        _total = np.sum(num)
        self.weight = torch.tensor([(_total - i) / _total for i in num]).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        ce_loss = self.ce_loss(input, target)
        ce_loss += ce_loss * self.weight[target]  # weight < 1

        if self.reduction == 'mean':
            return ce_loss.mean()
        elif self.reduction == 'sum':
            return ce_loss.sum()
        else:
            return ce_loss

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = 2  # focal loss num_classes = 2, 原论文也只是二分类
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        logits = input
        probs = torch.softmax(logits, dim=1)
        if target.shape[-1] == 1 or target.dim() == 1:  # (BatchSize, 1) or (BatchSize, )
            label = F.one_hot(target.to(torch.int64), num_classes=self.num_classes).float()
        else:
            target = target.squeeze()
            if target.shape[-1] == self.num_classes and target.dim() == 2:  # (BatchSize, numclasses)
                label = target.float()
            else:
                raise ValueError("label shape not supported")
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalLoss_with_LabelSmoothing(FocalLoss):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', smoothing=0.06):
        super(FocalLoss_with_LabelSmoothing, self).__init__(alpha, gamma, reduction)
        self.smoothing = smoothing
    
    def forward(self, input, target):
        target = F.one_hot(target.to(torch.int64), num_classes=self.num_classes).float()
        target = (1 - self.smoothing) * target + self.smoothing / 2
        return super().forward(input, target)


def L1_regular(model, factor=1e-6):
    # 设置成1e-2都会导致loss为5000+
    l1_loss = torch.tensor(0., requires_grad=True).cuda()
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + torch.norm(param, 1)
    return factor * l1_loss
    
def L2_regular(model, factor=1e-6):
    l2_loss = torch.tensor(0., requires_grad=True).cuda()
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + torch.norm(param, 2)
    return factor * l2_loss
