# 每个患者的随机染色200张并保存
import pickle as pkl
import numpy as np
import os
import random
import json
import tqdm
import multiprocessing as mp
from staintools import ReinhardColorNormalizer, StainNormalizer


def preprocess(npy_path, pid):
    normalizer_root = "./Data/Normalizers"
    npy_data = np.load(npy_path)
    print(f"Processing {pid} with shape {npy_data.shape}")
    for i in range(npy_data.shape[0]):
        image = npy_data[i, :, :, :]
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        image = np.ascontiguousarray(image)
        for method in ['macenko', 'vahadane', 'reinhard']:
            try:
                normalizer_dir = os.path.join(normalizer_root, method, str(pid))
                if not os.path.exists(normalizer_dir):
                    os.makedirs(normalizer_dir)
                if os.path.exists(os.path.join(normalizer_dir, f'normalizer_{i}.pkl')):
                    continue
                if method == 'reinhard':
                    normalizer = ReinhardColorNormalizer()
                elif method == 'macenko':
                    normalizer = StainNormalizer(method)
                elif method == 'vahadane':
                    normalizer = StainNormalizer(method)
                normalizer.fit(image)
                with open(os.path.join(normalizer_dir, f'normalizer_{i}.pkl'), 'wb') as f:
                    pkl.dump(normalizer, f)
            except Exception as e:
                print(f"Error processing {pid} with shape {npy_data.shape} at {i} {method}: {e}")
                continue


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")

    # fixed seed
    random.seed(2024)
    np.random.seed(2024)
    
     # load splits, only train set is used to calculate normalizers
    with open("./Data/split_seed=2024.json", 'r') as f:
        train_pids = json.load(f)['train']

    # load info
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        pid_to_npy_path = {item['pid']: item['path'] for item in info}
    
    tasks = []
    for pid in train_pids:
        tasks.append((pid_to_npy_path[pid], pid))
    
    print(f"Total {len(tasks)} tasks")
    with mp.Pool(8) as p:
        p.starmap(preprocess, tasks)