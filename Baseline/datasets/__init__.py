from utils.config import parse_arguments
from datasets.data_utils import *
from torch.utils.data import DataLoader
from datasets.default import *
from datasets.stain import *
from datasets.data_sampler import BalancedBatchSampler, DistributedBalancedBatchSampler
from datasets.data_features import MyFeatDataset 

def GetDataLoader(fold, mean_std=None, args=None, test_mode=False):
    args = parse_arguments() if args is None else args
    if args.input_feats:
        return GetTVTFeatureLoader(mean_std, args, test_mode)
    if args.train_mode == "TVT":
        return GetTVTDataLoader(mean_std, args, test_mode)
    else:
        return GetCVDataLoader(fold, mean_std, args, test_mode)


def GetTVTFeatureLoader(mean_std=None, args=None, test_mode=False):
    train_set = MyFeatDataset(0, "train", args.data_type, mean_std, test_mode, args)
    valid_set = MyFeatDataset(0, "valid", args.data_type, mean_std, True, args)
    test_set = MyFeatDataset(0, "test", args.data_type, mean_std, True, args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    
    return train_loader, valid_loader, test_loader


def GetTVTDataLoader(mean_std=None, args=None, test_mode=False):
    
    train_set = GetDataset(0, "train", args.data_type, mean_std, test_mode, args)
    mean_std = train_set.mean_std
    valid_set = GetDataset(0, "valid", args.data_type, mean_std, True, args)
    test_set = GetDataset(0, "test", args.data_type, mean_std, True, args)
    
    if test_mode:
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
            num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    else:
        if args.use_ddp:
            num_gpus = torch.cuda.device_count()
            assert args.batch_size % num_gpus == 0, "Batch size should be divisible by number of GPUs"
            train_loader = DataLoader(train_set, batch_size=args.batch_size // num_gpus,
                sampler=DistributedBalancedBatchSampler(train_set), num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
            print("Using DDP with batch size: ", args.batch_size // num_gpus)
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, 
                sampler=BalancedBatchSampler(train_set), num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
            print("Using batch size: ", args.batch_size)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)

    return train_loader, valid_loader, test_loader

def GetCVDataLoader(fold, mean_std=None, args=None, test_mode=False):
    
    train_set = GetDataset(fold, "train", args.data_type, mean_std, test_mode, args)
    mean_std = train_set.mean_std
    valid_set = GetDataset(fold, "valid", args.data_type, mean_std, True, args)
    
    if args.use_ddp:
        num_gpus = torch.cuda.device_count()
        assert args.batch_size % num_gpus == 0, "Batch size should be divisible by number of GPUs"
        train_loader = DataLoader(train_set, batch_size=args.batch_size // num_gpus,
            sampler=DistributedBalancedBatchSampler(train_set), num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
        print("Using DDP with batch size: ", args.batch_size // num_gpus)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, 
            sampler=BalancedBatchSampler(train_set), num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
        print("Using batch size: ", args.batch_size)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)

    return train_loader, valid_loader, None



def GetDataset(currentfold=0, type="train", data_type="ALL", mean_std=None, test_mode=False, args=None):
    assert args is not None, "Please provide args!"
    # stain dataset when passing augment method
    if args.augment_method is not None:
        assert args.augment_method in ['macenko', 'reinhard', 'vahadane']
        print(f"Using Stain Dataset method = {args.augment_method}!")
        return StainDataset(args.augment_method, currentfold, type, data_type, mean_std, test_mode, args)
    else:
        print("Using Default Dataset!")
        return MyBaseDataset(currentfold, type, data_type, mean_std, test_mode, args)

