# 每个患者的随机染色200张并保存
import pickle as pkl
import numpy as np
import os
import random
import json
import tqdm
import multiprocessing as mp



def norm_data(source_path, pid_result_root, ref_pid, cluster):
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
    np.save(os.path.join(pid_result_root, f'ref_pid={ref_pid}.npy'), norm_data)
                

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
    
    # fixed seed
    random.seed(2024)
    np.random.seed(2024)
    
     # load splits, only train set is used to calculate normalizers
    with open("./Data/split_seed=2024.json", 'r') as f:
        pids = json.load(f)['train']

    # load info
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        info = [x for x in info if x['pid'] in pids]

    # load cluster info
    with open("./Data/32Cluster.json", 'r') as f:
        cluster_info = json.load(f)
        
    results_root = "./Data/vahadane_images"

    tasks = []
    for item in info:
        pid_result_root = os.path.join(results_root, str(item['pid']))
        os.makedirs(pid_result_root, exist_ok=True)
        for c in range(32):
            # random select an normalizer from norm_dir in cluster c
            while True:
                ref_pid = random.choice(cluster_info[str(c)])
                if ref_pid != item['pid']:
                    break
            # set tasks
            if not os.path.exists(os.path.join(pid_result_root, f'ref_pid={ref_pid}.npy')):
                tasks.append((item['path'], pid_result_root, ref_pid, c))

    with mp.Pool(8) as p:
        p.starmap(norm_data, tasks)