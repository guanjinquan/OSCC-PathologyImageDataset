import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import json
import numpy as np

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
    Cluster_NUM = 8
    
    with open("./Data/split_seed=2024.json", 'r') as f:
        pids = json.load(f)['train']
    
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        info = [x for x in info if x['pid'] in pids]
    
    # get 512 embedding from 6x histogram embedding
    pca = PCA(n_components=512)
    embeds = []
    pid_ord = []
    for item in info:
        data = np.load(item['path'])
        pat_embeds = []
        for i in range(data.shape[0]):
            image = data[i, :, :, :]
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            image = np.ascontiguousarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()
            pat_embeds.append(hist)
        pat_embeds = np.concatenate(pat_embeds, axis=0).reshape(1, -1)
        embeds.append(pat_embeds)
        pid_ord.append(item['pid'])
    embeds = np.concatenate(embeds, axis=0)
    print(embeds.shape)
    pca.fit(embeds)
    embeds = pca.transform(embeds)
    
    kmeans = KMeans(n_clusters=Cluster_NUM, random_state=2024)
    kmeans.fit(embeds)
    labels = kmeans.labels_
    clusters_idx = kmeans.cluster_centers_
    
    # print cluster center pid
    cluster_pid = {}
    for i in range(Cluster_NUM):
        cluster_pid[i] = []
    for i, pid in enumerate(pid_ord):
        cluster_pid[labels[i]].append(pid)
    
    with open(f"./{Cluster_NUM}Cluster.json", 'w') as f:
        json.dump(cluster_pid, f)
    
