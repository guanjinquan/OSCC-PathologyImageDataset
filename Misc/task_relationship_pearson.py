# pearson
import numpy as np
import scipy.stats as stats
import json
import os


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/../")
    
    with open("./Data/all_metadata.json") as f:
        metadata = json.load(f)["datainfo"]
        
    task_labels = {
        "REC": [],
        "LNM": [],
        "TD": [],
        "TI": [],
        "CE": [],
        "PI": [],
    }
    
    for item in metadata:
        for key in task_labels:
            if item[key] != "None":
                task_labels[key].append(item[key])
            else:
                task_labels[key].append(-1)
    
    # pearson
    table = []
    for key in task_labels:
        pearson = []
        for key2 in task_labels:
            pearson.append(np.round(stats.pearsonr(task_labels[key], task_labels[key2])[0], 4))
        table.append(pearson)
    
    print("Pearson correlation coefficient")
    print("REC LNM TD TI CE PI")
    for i in range(len(table)):
        print(table[i])
        
    