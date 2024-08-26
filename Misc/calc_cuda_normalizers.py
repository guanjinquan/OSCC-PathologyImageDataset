from torchvahadane import TorchVahadaneNormalizer
import json
import numpy as np
import os
import time
import pickle as pkl

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
    
    # load splits, only train set is used to calculate normalizers
    with open("./split_seed=2024.json", 'r') as f:
        pids = json.load(f)['train']

    # load info
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        info = [x for x in info if x['pid'] in pids]   # load splits, only train set is used to calculate normalizers

    TEST_LOAD_TIME = True
    
    # calculate normalizers
    results_root = "./Data/cuda_normalizers"
    for item in info:
        print(item['pid'])
        data = np.load(item['path'])
        pid_result_root = os.path.join(results_root, str(item['pid']))
        os.makedirs(pid_result_root, exist_ok=True)
        for i in range(data.shape[0]):
            image = data[i, :, :, :]
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            reference_image = np.ascontiguousarray(image)
            normalizer = TorchVahadaneNormalizer(device='cuda', staintools_estimate=True)
            normalizer.fit(reference_image)
            with open(os.path.join(pid_result_root, f'normalizer_{i}.pkl'), 'wb') as f:
                pkl.dump(normalizer, f)
            
            if TEST_LOAD_TIME:
                TEST_LOAD_TIME = False
                start = time.time()
                with open(os.path.join(pid_result_root, f'normalizer_{i}.pkl'), 'rb') as f:
                    normalizer = pkl.load(f)
                end = time.time()
                print("Load time cost: ", end - start)
            
            
    