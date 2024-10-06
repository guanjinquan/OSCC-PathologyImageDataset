# 每个患者的随机染色200张并保存
import pickle as pkl
import numpy as np
import os
import random
import json
from PIL import Image


def normalize(source_path, ref_pid, cluster):
    print(f"Processing {source_path} with ref_pid={ref_pid} cluster = {cluster}")
    data = np.load(source_path)
    normalizer_root = "./Data/cuda_normalizers"
    normalizer_dir = os.path.join(normalizer_root, str(ref_pid))
    norm_data = []
    
    for i in range(data.shape[0]):
        image = data[i, :, :, :]
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        with open(os.path.join(normalizer_dir, f'normalizer_{i}.pkl'), 'rb') as f:
            normalizer = pkl.load(f)
        image = normalizer.transform(image)
        
        image = image.cpu().numpy()
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        norm_data.append(image)  # [C, H, W]
        
    norm_data = np.stack(norm_data, axis=0)
    return norm_data
    

def plot_image(data, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i in range(data.shape[0]):
        img_i = data[i, :, :, :]
        img_i = np.transpose(img_i, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(img_i).save(os.path.join(save_path, f"{i}.png"))
        

if __name__ == "__main__":
    source_pid = 1093516
    Cluster_NUM = 8
    os.chdir(os.path.dirname(__file__) + "/..")
    
     # load splits, only train set is used to calculate normalizers
    with open("./Data/split_seed=2024.json", 'r') as f:
        pids = json.load(f)['train']

    # load info
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        info = [x for x in info if x['pid'] in pids]

    # load cluster info
    with open(f"./Data/{Cluster_NUM}Cluster.json", 'r') as f:
        cluster_info = json.load(f)
    
    # stain normalize for each cluster
    visualize_temp_root = "./Data/visualize_temp_images"
    os.makedirs(visualize_temp_root, exist_ok=True)
    
    plot_image(np.load(f"./Data/NpyData/{source_pid}.npy"), os.path.join(visualize_temp_root, "source"))
    
    for c in range(Cluster_NUM):
        # random select an normalizer from norm_dir in cluster c
        while True:
            ref_pid = random.choice(cluster_info[str(c)])
            if ref_pid != source_pid:
                break
        
        c_data = normalize(f"./Data/NpyData/{source_pid}.npy", ref_pid, c)
        plot_image(c_data, os.path.join(visualize_temp_root, f"cluster_{c}"))
        
            
    
