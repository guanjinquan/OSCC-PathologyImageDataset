import json
from collections import Counter
import os

os.chdir(os.path.dirname(__file__) + "/../")

Tasks = [
    "REC",
    "LNM",
    "TD",
    "TI",
    "CE",
    "PI"
]


with open("./Data/all_metadata.json", 'r') as f:
    metadata = json.load(f)['datainfo']
    
with open("./Data/split_seed=2024.json", 'r') as f:
    split = json.load(f)
    
train = split['train']
val = split['valid']
test = split['test']


for task in Tasks:
    print(task)
    for mode in ['train', 'valid', 'test']:
        print(mode)
        arrlist = []
        for item in metadata:
            pid = item['pid']
            if pid in split[mode]:
                arrlist.append(item[task])
        print(Counter(arrlist))
    print()