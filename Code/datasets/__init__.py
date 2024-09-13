from utils.config import parse_arguments
from datasets.data_utils import *
from torch.utils.data import DataLoader
from datasets.default import *
from datasets.vahadane import *
from datasets.data_sampler import BalancedBatchSampler


def GetDataLoader(fold, mean_std=None, args=None, test_mode=False):
    args = parse_arguments() if args is None else args
    if args.train_mode == "TVT":
        return GetTVTDataLoader(mean_std, args, test_mode)
    else:
        return GetCVDataLoader(fold, mean_std, args, test_mode)


def GetTVTDataLoader(mean_std=None, args=None, test_mode=False):
    
    train_set = GetDataset(0, "train", "ALL", mean_std, test_mode, args)
    mean_std = train_set.mean_std
    valid_set = GetDataset(0, "valid", "ALL", mean_std, True, args)
    test_set = GetDataset(0, "test", "ALL", mean_std, True, args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, 
        sampler=BalancedBatchSampler(train_set), num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)

    return train_loader, valid_loader, test_loader

def GetCVDataLoader(fold, mean_std=None, args=None, test_mode=False):
    
    train_set = GetDataset(fold, "train", "ALL", mean_std, test_mode, args)
    mean_std = train_set.mean_std
    valid_set = GetDataset(fold, "valid", "ALL", mean_std, True, args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, 
        sampler=BalancedBatchSampler(train_set), num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=4, pin_memory=True, collate_fn=collate_fn_ensemble)

    return train_loader, valid_loader, None



def GetDataset(currentfold=0, type="train", data_type="ALL", mean_std=None, test_mode=False, args=None):
    assert args is not None, "Please provide args!"
    if args.augment_method == "vahadane":
        print("Using VahadaneDataset!")
        return VahadaneDataset(currentfold, type, data_type, mean_std, test_mode, args)
    else:
        print("Using VanillaDataset!")
        return MyBaseDataset(currentfold, type, data_type, mean_std, test_mode, args)

