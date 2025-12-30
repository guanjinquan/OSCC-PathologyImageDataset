import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    norm = False
    model_selection = "vit_small_p16_pathology"
    class Args:
        model = model_selection
        freezed_backbone = True
        # img_size = (1944, 2592)
        img_size = (512, 512)

    model, dim = get_backbone_and_embed_dim(Args())
    model = model.cuda()


    # exract features from a folder of images
    # folder_path = "../Data/NpyData_ORI_SIZE"
    # output_path = f"../Data/Features_Norm_ORI_SIZE_{model_selection}"
    # folder_path = "../Data/NpyData"
    # output_path = f"../Data/Features_Norm_{model_selection}"
    folder_path = "../Data/NpyData"
    output_path = f"../Data/Features_{model_selection}"
    os.makedirs(output_path, exist_ok=True)

    mead_std = ([175.14728804175988, 110.57123792228117, 176.73598615775617], [21.239463551725915, 39.15991384752335, 10.99100631656543])

    for file in tqdm.tqdm(os.listdir(folder_path)):
        result_file = os.path.join(output_path, file.replace(".npy", ".npy"))
        if os.path.exists(result_file):
            print(f"File {result_file} already exists, skipping.")
            continue

        data = np.load(os.path.join(folder_path, file))
        features = []
        for i in range(data.shape[0]):
            img = data[i].astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0)

            if norm:
                for j in range(3):
                    img[:, j, :, :] = (img[:, j, :, :] - mead_std[0][j]) / mead_std[1][j]

            with torch.no_grad():
                feature = model(img.cuda()).cpu().numpy()
            features.append(feature)
        features = np.concatenate(features, axis=0)
        print("SHAPE = ", features.shape)
        np.save(result_file, features)

