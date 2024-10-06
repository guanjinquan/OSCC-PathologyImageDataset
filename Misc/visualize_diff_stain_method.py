import os
import sys
import random
import numpy as np
import json
from PIL import Image
from staintools import StainNormalizer, ReinhardColorNormalizer


def plot_images(images, path_visual_root):
    os.makedirs(path_visual_root, exist_ok=True)
    for i in range(images.shape[0]):
        img_i = images[i, :, :, :]
        img_i = np.transpose(img_i, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(img_i).save(os.path.join(path_visual_root, f"{i}.png"))


def normalizer(method, source, target):
    ret = []
    for i in range(source.shape[0]):
        source_img = source[i, :, :, :]
        target_img = target[i, :, :, :]
        source_img = np.ascontiguousarray(np.transpose(source_img, (1, 2, 0))).astype(np.uint8)
        target_img = np.ascontiguousarray(np.transpose(target_img, (1, 2, 0))).astype(np.uint8)
        if method == 'vahadane':
            normalizer = StainNormalizer(method)
        elif method == 'macenko':
            normalizer = StainNormalizer(method)
        elif method == 'reinhard':
            normalizer = ReinhardColorNormalizer()
        normalizer.fit(target_img)
        source_img_normalized = normalizer.transform(source_img)
        ret.append(np.transpose(source_img_normalized, (2, 0, 1)))
    return np.array(ret)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
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

    source_pid = 1093516   # random.choice(list(pids))
    source_path = pid_to_path[source_pid]
    source_img = np.load(source_path)
    plot_images(source_img, f"./Data/visualize_diff_stain_images/source")
    
    for c in range(Cluster_NUM):
        train_cluster_pids = set(cluster_info[str(c)]) & set(pids)
        target_pid = random.choice(list(train_cluster_pids))
        target_path = pid_to_path[target_pid]
        target_img = np.load(target_path)
        plot_images(target_img, f"./Data/visualize_diff_stain_images/cluster_{c}/target")
        for method in ['vahadane', 'macenko', 'reinhard']:
            print("Working on : ", method, c, source_pid, target_pid)
            source_img_normalized = normalizer(method, source_img, target_img)
            path_visual_root = f"./Data/visualize_diff_stain_images/cluster_{c}/{method}_stain"
            plot_images(source_img_normalized, path_visual_root)