import pickle as pkl
import numpy as np
import os
import random
import multiprocessing as mp
import json
from staintools import LuminosityStandardizer
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer
from torchvahadane import TorchVahadaneNormalizer
import torch

"""
this script has the following shortcomings in macenko normalizer:
    1. not all target images is standardized, if the standardization fails, the original image is used
    2. the beta parameter in Macenko normalizer is not always 0.15 because of the brightness standardization
    3. even though we specially solve the exception in Macenko normalizer using the above method, 
        the normalizer may still fail to fit the target image sometimes
"""


def worker(method, target_img, i, normalizer_dir):
    assert type(target_img) == np.ndarray
    if method == 'vahadane':
        normalizer = TorchVahadaneNormalizer(device='cuda', staintools_estimate=True)
        normalizer.fit(target_img)
    elif method == 'macenko':
        target_img = torch.from_numpy(target_img).permute(2, 0, 1)
        normalizer = TorchMacenkoNormalizer()
        try:
            normalizer.fit(target_img)
        except Exception as e:
            # sometimes the beta is too large which will cause exception
            normalizer.fit(target_img, beta=0.05)  
    elif method == 'reinhard':
        target_img = torch.from_numpy(target_img).permute(2, 0, 1)
        normalizer = TorchReinhardNormalizer()
        normalizer.fit(target_img)
    with open(os.path.join(normalizer_dir, f'normalizer_{i}.pkl'), 'wb') as f:
        pkl.dump(normalizer, f)


def preprocess(npy_path, pid):
    normalizer_root = "./Data/CudaNormalizers"
    npy_data = np.load(npy_path)
    print(f"Processing {pid} with shape {npy_data.shape}")
    for i in range(npy_data.shape[0]):
        image = npy_data[i, :, :, :]
        image = np.transpose(image, (1, 2, 0))
        assert image.min() >= 0 and image.max() <= 255
        image = image.astype(np.uint8)
        image = np.asfortranarray(image)
        for method in ['macenko', 'vahadane', 'reinhard']:
            try:
                normalizer_dir = os.path.join(normalizer_root, method, str(pid))
                if not os.path.exists(normalizer_dir):
                    os.makedirs(normalizer_dir)
                if os.path.exists(os.path.join(normalizer_dir, f'normalizer_{i}.pkl')):
                    print(f"Torch Normalizer for {pid} at {i} {method} already exists")
                    continue
                # try to standardize the image
                target_image = LuminosityStandardizer.standardize(image)  
                worker(method, target_image, i, normalizer_dir)
            except Exception as e:
                try:
                    # if standardization fails, try to no standardize
                    worker(method, image, i, normalizer_dir)
                except Exception as e:
                    # all fails, skip this normalizer
                    print(f"Error processing {pid} with shape {npy_data.shape} at {i} {method}: {e}")
                    continue

if __name__ == "__main__": 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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