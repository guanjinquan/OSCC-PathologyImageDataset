import torch
import numpy as np

def collate_fn_ensemble(input):  
    data = None
    target = {}
    idxs = []

    # Multi-view
    for x, y, i in input:  # new是新在没有0填充
        for k, v in y.items():
            target[k] = target.get(k, []) + (v if type(v) is list else [v])
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        data = torch.cat((data, x), dim=0) if data != None else x
        idxs.append(i)  # 需要获取病人id
        
    # data : tensor[number of images in batch, 3, H, W]
    # targget : {'task': ytrue, ...}
    # idxs : tensor[pid1, pid2, ...]
    data = data.cpu()
    for k, v in target.items():
        target[k] = torch.tensor(v, dtype=torch.long).cpu()
    idxs = torch.tensor(idxs, dtype=torch.long).cpu()
    return (data, target, idxs)
