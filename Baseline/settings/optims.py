from torch.optim import Adam, AdamW
import torch

# Optimizer参数:
# optimizer: 'Adam' / 'AdamW'
# backbone_lr: 0.0001
# learning_rate: 0.001
# weight_decay: 0.0001

# Scheduler参数:
# scheduler: 'CosineAnnealingLR' / 'CosineAnnealingLR_warmup' / 'OneCycleLR'
# num_epochs: 100


def GetOptimizer(args, model):
    if args.optimizer == "AdamW" and args.freezed_backbone:
        print("Using AdamW optimizer with freezed backbone")
        return AdamW([
            {'params': model.get_others_params(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ])
    
    if args.optimizer == 'Adam':
        return Adam([
            {'params': model.get_backbone_params(), 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
            {'params': model.get_others_params(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ])
    elif args.optimizer == 'AdamW':
        return AdamW([
            {'params': model.get_backbone_params(), 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
            {'params': model.get_others_params(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ])
    elif args.optimizer == 'SGD':
        print("SGD optimizer")
        return torch.optim.SGD([
            {'params': model.get_backbone_params(), 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
            {'params': model.get_others_params(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ], momentum=0.9)
    else:
        raise ValueError("optimizer not supported")
    
def GetScheduler(args, optim):
    if args.scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs, eta_min=1e-9)
    elif args.scheduler == 'CosineAnnealingLR_warmup':
        assert args.num_epochs % 2 == 0, "num_epochs must be even"
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs // 2, eta_min=1e-9)
    elif args.scheduler == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
                optim, 
                max_lr=[args.backbone_lr, args.learning_rate], 
                epochs=args.num_epochs, 
                steps_per_epoch=1, 
                anneal_strategy='cos'
            )
    else:
        raise ValueError("scheduler not supported") 
