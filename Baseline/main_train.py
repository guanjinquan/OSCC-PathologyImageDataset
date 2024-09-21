import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import parse_arguments
import numpy as np
import random
from modules.trainer import Trainer
import wandb


if __name__ == '__main__':
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    os.chdir(os.path.dirname(__file__) + "/../")
    torch.multiprocessing.set_start_method('spawn')
    
    # 获取显卡显存大小
    memory_size = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    if memory_size < 20 or args.use_ddp:  # < 20 GB or ddp to set max_split_size_mb:128
        print("Memory size is less than 20GB, set PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:128 !!!", flush=True)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
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
        trainer = Trainer(fold=0, args=args)
        
        if trainer.local_rank == 0:
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
                    'data_type': args.data_type,
                },
                settings=wandb.Settings(_service_wait=300)
            )
            
        trainer.run()
        
        if trainer.local_rank == 0:
            wandb.finish()
    else:
        cv_list = eval(args.train_mode)
        for fold in cv_list:
            trainer = Trainer(fold=fold, args=args)
    
            if trainer.local_rank == 0:
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
                        'data_type': args.data_type,
                    },
                    settings=wandb.Settings(_service_wait=300)
                )
            
            trainer.run()
            
            if trainer.local_rank == 0:
                wandb.finish()

    