import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../Baseline")
sys.path.append(os.getcwd())
import torch
import os
import sys
from utils.config import parse_arguments
import numpy as np
import random
import wandb
from models import get_backbone_and_embed_dim
import os
import tqdm
import numpy as np
import wandb
import torch

if __name__ == "__main__":
    model_selection = "resnet50_imagenet"
    model_selection = "vit_small_p16_pathology"
    class Args:
        model = model_selection
        freezed_backbone = True
        img_size = (1944, 2592)

    model, dim = get_backbone_and_embed_dim(Args())
    model = model.cuda()


    # exract features from a folder of images
    folder_path = "../Data/NpyData_ORI_SIZE"
    output_path = f"../Data/Features_ORI_SIZE_{model_selection}"
    os.makedirs(output_path, exist_ok=True)

    for file in tqdm.tqdm(os.listdir(folder_path)):
        data = np.load(os.path.join(folder_path, file))
        features = []
        for i in range(data.shape[0]):
            img = data[i].astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0)
            with torch.no_grad():
                feature = model(img.cuda()).cpu().numpy()
            features.append(feature)
        features = np.concatenate(features, axis=0)
        print("SHAPE = ", features.shape)
        np.save(os.path.join(output_path, file.replace(".npy", ".npy")), features)

