import pickle as pkl
import numpy as np
import os
import random
import json
import tqdm
import multiprocessing as mp
from torchstain.torch.normalizers import TorchReinhardNormalizer, TorchMacenkoNormalizer
from torchvahadane import TorchVahadaneNormalizer
from staintools import LuminosityStandardizer, ReinhardColorNormalizer, StainNormalizer
import torch



def torch_worker(method, normalizer_dir, i, image_standardized):
    with open(os.path.join(normalizer_dir, f'normalizer_{i}.pkl'), 'rb') as f:
        normalizer = pkl.load(f)
    if method == 'vahadane':
        image = normalizer.transform(image_standardized)
    elif method == 'macenko':
        image = torch.from_numpy(image_standardized).permute(2, 0, 1)  # [C, H, W]
        image = normalizer.normalize(I=image)[0]
    elif method == 'reinhard':
        image = torch.from_numpy(image_standardized).permute(2, 0, 1)
        image = normalizer.normalize(I=image)    
    image_normed = np.transpose(image.cpu().numpy(), (2, 0, 1)).astype(np.float32)  # [C, H, W]
    return image_normed


def staintools_worker(method, ref_data, i, image_standardized):
    ref_image = ref_data[i, :, :, :]
    ref_image = np.transpose(ref_image, (1, 2, 0)).astype(np.uint8)
    ref_image = np.ascontiguousarray(ref_image)
    ref_image = LuminosityStandardizer.standardize(ref_image)
    if method == 'vahadane':
        normalizer = StainNormalizer(method='vahadane')
        normalizer.fit(ref_image)
        image_normed = normalizer.transform(image_standardized)
    elif method == 'macenko':
        normalizer = StainNormalizer(method='macenko')
        normalizer.fit(ref_image)
        image_normed = normalizer.transform(image_standardized)
    elif method == 'reinhard':
        normalizer = ReinhardColorNormalizer()
        normalizer.fit(ref_image)
        image_normed = normalizer.transform(image_standardized)
    image_normed = np.transpose(image_normed, (2, 0, 1)).astype(np.float32)  # [C, H, W]
    return image_normed


def normalize_data(method, ref_pid, source_pid, ref_path, source_path):
    print(f"Processing {source_path} with ref_pid={ref_pid} method = {method}")
    data = np.load(source_path)
    ref_data = None
    normalizer_dir = f"./Data/CudaNormalizers/{method}/{ref_pid}"
    pid_result_root = f"./Data/StainNormedData/{method}/{source_pid}"
    os.makedirs(pid_result_root, exist_ok=True)
    
    if os.path.exists(os.path.join(pid_result_root, f'ref_pid={ref_pid}.npy')):
        print(f"Already exists {source_path} with ref_pid={ref_pid}")
        return
    
    norm_data = []
    for i in range(data.shape[0]):
        image = data[i, :, :, :]
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        image = np.ascontiguousarray(image)
        image_standardize = LuminosityStandardizer.standardize(image)
        
        if os.path.exists(os.path.join(normalizer_dir, f'normalizer_{i}.pkl')):
            try:
                image_normed = torch_worker(method, normalizer_dir, i, image_standardize)
                with open('./Misc/prestain_log.txt', 'a') as f:
                    f.write(f"[OK] Success at {source_pid} with ref_pid={ref_pid}\n")
            except Exception as e:
                with open("./Misc/prestain_log.txt", 'a') as f:
                    f.write(f" - [Error-turn-to-staintools-worker] {e} at {source_pid} with ref_pid={ref_pid}\n")
                if ref_data is None:
                    ref_data = np.load(ref_path)
                try:
                    image_normed = staintools_worker(method, ref_data, i, image_standardize)
                    with open('./Misc/prestain_log.txt', 'a') as f:
                        f.write(f" - [OK-staintools-after_torch_error] But success at {source_pid} with ref_pid={ref_pid}\n")
                except Exception as e:
                    with open("./Misc/prestain_log.txt", 'a') as f:
                        f.write(f"    - [Error-torch-staintools-all-fails] {e} at {source_pid} with ref_pid={ref_pid}\n")
                    image_normed = np.transpose(image_standardize, (2, 0, 1)).astype(np.float32)
        else:
            if ref_data is None:
                ref_data = np.load(ref_path)
            try:
                image_normed = staintools_worker(method, ref_data, i, image_standardize)
                with open('./Misc/prestain_log.txt', 'a') as f:
                    f.write(f"[OK unfound cuda normalizer] But success at {source_pid} with ref_pid={ref_pid}\n")
            except Exception as e:
                image_normed = np.transpose(image_standardize, (2, 0, 1)).astype(np.float32)
                with open('./Misc/prestain_log.txt', 'a') as f:
                    f.write(f"[Error unfound cuda normalizer and staintools fail] {e} at {source_pid} with ref_pid={ref_pid}\n")
                    
        norm_data.append(image_normed)  # [C, H, W]
    norm_data = np.stack(norm_data, axis=0)  # [6, C, H, W]
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
                tasks.append((method, ref_pid, source_pid, pid_to_npy_path[ref_pid], pid_to_npy_path[source_pid]))
    
    print(f"Total {len(tasks)} tasks")
    with mp.Pool(8) as p:
        p.starmap(normalize_data, tasks)