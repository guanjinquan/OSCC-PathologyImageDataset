import os
import sys
import random
import numpy as np
import json
from PIL import Image


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    Cluster_NUM = 8
    
    # fixed seed
    # random.seed(2024)
    # np.random.seed(2024)
    
     # load splits, only train set is used to calculate normalizers
    with open("./Data/split_seed=2024.json", 'r') as f:
        pids = json.load(f)['train']

    # load info
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        pid_to_path = {x['pid']: x['path'] for x in info}

    # load cluster info
    with open(f"./Data/{Cluster_NUM}Cluster.json", 'r') as f:
        cluster_info = json.load(f)

    visualize_root = "./Data/visualize_images"
    os.makedirs(visualize_root, exist_ok=True)
    
    for cluster in range(Cluster_NUM):
        train_cluster_pids = set(cluster_info[str(cluster)]) & set(pids)
        pat_id = random.choice(list(train_cluster_pids))
        img_path = pid_to_path[pat_id]
        
        img = np.load(img_path)
        path_visual_root = os.path.join(visualize_root, str(cluster))
        os.makedirs(path_visual_root, exist_ok=True)
        for i in range(img.shape[0]):
            img_i = img[i, :, :, :]
            img_i = np.transpose(img_i, (1, 2, 0)).astype(np.uint8)
            Image.fromarray(img_i).save(os.path.join(path_visual_root, f"{i}.png"))

