import torch
import os
import pickle as pkl

def save_trainer(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pkl.dump(obj, f)
    
def load_trainer(path):
    if not os.path.exists(path):
        raise ValueError("Error: args and path are None.")
    with open(path, 'rb') as f:
        checkpoint = pkl.load(f)
    return checkpoint

def save_model(model, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict(),  'epoch': epoch}, path)
    
def load_model(path):
    if not os.path.exists(path):
        raise ValueError(f"Error: {path} not exists.")
    checkpoint = torch.load(path, map_location='cpu')
    
    # 更新 task -> tasks
    for name in list(checkpoint['model'].keys()):
        if "task" in name and "tasks" not in name:
            new_name = name.replace("task", "tasks")
            checkpoint['model'][new_name] = checkpoint['model'][name]
            del checkpoint['model'][name]
    
    print("Loading model : epoch = ", checkpoint.get('epoch', "Unknown"))
    return checkpoint   # 只加载权重
