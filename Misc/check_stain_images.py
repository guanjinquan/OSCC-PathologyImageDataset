import os
import json
import numpy as np


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
    
    # load splits, only train set is used to calculate normalizers
    with open("./Data/split_seed=2024.json", 'r') as f:
        pids = json.load(f)['train']

    stain_root = "./Data/StainNormedData"

    for method in ['macenko', 'vahadane', 'reinhard']:
        for pid in pids:
            if not os.path.exists(f"{stain_root}/{method}/{pid}"):
                print(f"[no such dir for patient] error on patient = {pid}.")
                continue
            length = len(os.listdir(f"{stain_root}/{method}/{pid}"))
            if length != 2:
                print(f"[error] error on patient = {pid}, length = {length}.")
                continue
            for file in os.listdir(f"{stain_root}/{method}/{pid}"):
                try:
                    np.load(f"{stain_root}/{method}/{pid}/{file}")
                except Exception as e:
                    print(f" - [error] error on patient = {pid}, file = {file}.")
                