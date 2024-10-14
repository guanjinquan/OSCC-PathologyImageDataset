import os
import sys
import random
import numpy as np
import json
from PIL import Image
from torchstain.torch.normalizers import TorchReinhardNormalizer, TorchMacenkoNormalizer
from torchvahadane import TorchVahadaneNormalizer
from staintools import LuminosityStandardizer, ReinhardColorNormalizer, StainNormalizer
import torch
from matplotlib import pyplot as plt


def histogram_normalizer(source, target):
    # 1.60 彩色图像的直方图匹配
    img = source  # flags=1 读取为彩色图像
    imgRef = target  # 匹配模板图像 (matching template)

    _, _, channel = img.shape
    imgOut = np.zeros_like(img)
    for i in range(channel):
        histImg, _ = np.histogram(img[:,:,i], 256)  # 计算原始图像直方图
        histRef, _ = np.histogram(imgRef[:,:,i], 256)  # 计算匹配模板直方图
        cdfImg = np.cumsum(histImg)  # 计算原始图像累积分布函数 CDF
        cdfRef = np.cumsum(histRef)  # 计算匹配模板累积分布函数 CDF
        for j in range(256):
            tmp = abs(cdfImg[j] - cdfRef)
            tmp = tmp.tolist()
            index = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
            imgOut[:,:,i][img[:,:,i]==j] = index
    return imgOut


def normalizer(method, source, target, standardized=True):
    ret = []
    for i in range(source.shape[0]):
        try:
            source_img = source[i, :, :, :]
            target_img = target[i, :, :, :]
            source_img = np.transpose(source_img, (1, 2, 0)).astype(np.uint8)
            target_img = np.transpose(target_img, (1, 2, 0)).astype(np.uint8)
            source_img = np.ascontiguousarray(source_img)
            target_img = np.ascontiguousarray(target_img)
            if standardized:
                target_img = LuminosityStandardizer.standardize(target_img)  # target img unneed to standardize``
            source_img = LuminosityStandardizer.standardize(source_img)
            if method == 'vahadane':
                normalizer = StainNormalizer(method='vahadane')
                normalizer.fit(target_img)
                source_img_normalized = normalizer.transform(source_img)
            elif method == 'macenko':
                normalizer = StainNormalizer(method='macenko')
                normalizer.fit(target_img)
                source_img_normalized = normalizer.transform(source_img)
            elif method == 'reinhard':
                normalizer = ReinhardColorNormalizer()
                normalizer.fit(target_img)
                source_img_normalized = normalizer.transform(source_img)
            ret.append(np.transpose(source_img_normalized, (2, 0, 1)))
        except Exception as e:
            print("Error: ", e)
            source_img = source[i, :, :, :]
            target_img = target[i, :, :, :]
            source_img = np.ascontiguousarray(np.transpose(source_img, (1, 2, 0))).astype(np.uint8)
            target_img = np.ascontiguousarray(np.transpose(target_img, (1, 2, 0))).astype(np.uint8)
            if standardized:
                target_img = LuminosityStandardizer.standardize(target_img)  # target img unneed to standardize``
            source_img = LuminosityStandardizer.standardize(source_img)
            if method == 'vahadane':
                normalizer = TorchVahadaneNormalizer(device='cuda', staintools_estimate=True)
                normalizer.fit(target_img)
                source_img_normalized = normalizer.transform(source_img)
            elif method == 'macenko':
                source_img = torch.from_numpy(source_img).permute(2, 0, 1)
                target_img = torch.from_numpy(target_img).permute(2, 0, 1)
                normalizer = TorchMacenkoNormalizer()
                normalizer.fit(target_img)
                source_img_normalized = normalizer.normalize(I=source_img)[0]
            elif method == 'reinhard':
                source_img = torch.from_numpy(source_img).permute(2, 0, 1)
                target_img = torch.from_numpy(target_img).permute(2, 0, 1)
                normalizer = TorchReinhardNormalizer()
                normalizer.fit(target_img)
                source_img_normalized = normalizer.normalize(I=source_img)
            ret.append(np.transpose(source_img_normalized.cpu().numpy(), (2, 0, 1)))
    return np.array(ret)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
    Cluster_NUM = 8
    
    # fixed seed
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    
     # load splits, only train set is used to calculate normalizers
    with open("./Data/split_seed=2024.json", 'r') as f:
        pids = json.load(f)['train']

    # load info
    with open("./Data/all_metadata.json", 'r') as f:
        info = json.load(f)['datainfo']
        pid_to_path = {x['pid']: x['path'] for x in info}

    source_pid = random.choice(list(pids))
    source_path = pid_to_path[source_pid]
    source_img = np.load(source_path)
    
    target_pid = random.choice(list(set(pids) - set([source_pid])))
    target_path = pid_to_path[target_pid]
    target_img = np.load(target_path)
    
    print("PIDS ; ", source_pid, target_pid)
    
    # plot (6 * 4) image
    plt.cla()
    fig, axs = plt.subplots(6, 5, figsize=(20, 20))
    for i in range(6):
        axs[i, 0].imshow(np.transpose(source_img[i, :, :, :], (1, 2, 0)).astype(np.uint8))
        axs[i, 0].axis('off')
    axs[0, 0].set_title('Source')
    
    for i in range(6):
        axs[i, 1].imshow(np.transpose(target_img[i, :, :, :], (1, 2, 0)).astype(np.uint8))
        axs[i, 1].axis('off')
    axs[0, 1].set_title('Reference')
    
    for i, method in enumerate(['macenko', 'vahadane', 'reinhard'], start=2):
        source_img_normalized = normalizer(method, source_img, target_img)
        for j in range(6):
            axs[j, i].imshow(np.transpose(source_img_normalized[j, :, :, :], (1, 2, 0)).astype(np.uint8))
            axs[j, i].axis('off')
        axs[0, i].set_title(method)

    plt.legend()
    plt.savefig(f"./Data/visualize_diff_stain_method.png")