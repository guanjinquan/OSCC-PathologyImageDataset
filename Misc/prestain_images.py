import pickle as pkl
import numpy as np
import os
import random
import json
import tqdm
import multiprocessing as mp


def normalize_data(method, ref_pid, source_pid, source_path):
    print(f"Processing {source_path} with ref_pid={ref_pid} method = {method}")
    data = np.load(source_path)
    normalizer_dir = f"./Data/Normalizers/{method}/{ref_pid}"
    pid_result_root = f"./Data/NormedData/{method}/{source_pid}"
    
    norm_data = []
    for i in range(data.shape[0]):
        image = data[i, :, :, :]
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        with open(os.path.join(normalizer_dir, f'normalizer_{i}.pkl'), 'rb') as f:
            normalizer = pkl.load(f)
        image = normalizer(image)
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
        train_pids = json.load(f)['train']

    # load info
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        pid_to_npy_path = {item['pid']: item['path'] for item in info}
    
    tasks = []
    for method in ['macenko', 'vahadane', 'reinhard']:
        for source_pid in train_pids:
            ref_list = []
            while len(ref_list) < 2:
                ref_pid = random.choice(list(set(train_pids) - set([source_pid]) - set(ref_list)))
                ref_list.append(ref_pid)
            for ref_pid in ref_list:
                tasks.append((method, ref_pid, source_pid, pid_to_npy_path[source_pid]))
    
    print(f"Total {len(tasks)} tasks")
    with mp.Pool(8) as p:
        p.starmap(normalize_data, tasks)