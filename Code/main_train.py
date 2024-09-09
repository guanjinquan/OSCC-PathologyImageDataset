import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import parse_arguments
import numpy as np
import random
from trainer import Trainer
import wandb


if __name__ == '__main__':
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    os.chdir(os.path.dirname(__file__) + "/../")
    torch.multiprocessing.set_start_method('spawn')
    
    
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
        wandb.init(
            project="OSCC-PathoCls",
            name=f"{args.model}-{args.runs_id}",
            config={
                'batch_size': args.batch_size * args.acc_step,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'backbone_lr': args.backbone_lr,
                'weight_decay': args.weight_decay,
                'backbones': args.model,
                'use_task': args.use_tasks,
                'optimizer': args.optimizer,
                'scheduler': args.scheduler,
                'seed': args.seed,
                'data_augment_method': args.augment_method,
                'stain_prob': args.stain_prob,
            },
            settings=wandb.Settings(_service_wait=300)
        )
        trainer = Trainer(fold=0, args=args)
        trainer.run()
        wandb.finish()
    else:
        cv_list = eval(args.train_mode)
        for fold in cv_list:
            wandb.init(
                project="OSCC-PathoCls",
                name=f"{args.model}-{args.runs_id}-{fold}",
                config={
                    'batch_size': args.batch_size *  args.acc_step,
                    'num_epochs': args.num_epochs,
                    'learning_rate': args.learning_rate,
                    'backbone_lr': args.backbone_lr,
                    'weight_decay': args.weight_decay,
                    'backbones': args.model,
                    'use_task': args.use_tasks,
                    'optimizer': args.optimizer,
                    'scheduler': args.scheduler,
                    'seed': args.seed,
                    'data_augment_method': args.augment_method,
                    'stain_prob': args.stain_prob,
                },
                settings=wandb.Settings(_service_wait=300)
            )
            trainer = Trainer(fold=fold, args=args)
            trainer.run()
            wandb.finish()

    