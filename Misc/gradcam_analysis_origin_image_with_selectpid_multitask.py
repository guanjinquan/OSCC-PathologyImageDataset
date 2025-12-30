import os
import sys
sys.path.append(os.path.dirname(__file__) + '/../Baseline')
sys.path.append(os.path.dirname(__file__) + '/../')
import torch.nn as nn
import cv2
import numpy as np
import torch
from Baseline.models import GetModel
from Baseline.oldmodels import GetModel as GetOldModel
from Baseline.datasets import GetDataLoader
from Baseline.utils import parse_arguments, load_model
import random
import glob
from collections import defaultdict
import itertools
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

# from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
# from pytorch_grad_cam.ablation_layer import AblationLayerVit


def reshape_transform(tensor, height=32, width=32):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class MultiTaskModel(nn.Module):
    def __init__(self, model):
        super(MultiTaskModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return [x]

    def loss(self, x, y):
        a, _ = self.model.loss(x, y)
        return a


def work(pth_path, pid_list):
    args = parse_arguments()
    args.batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    args.gpu_id = gpu_id
    
    args.split_filename = "split_seed=2024.json"
    args.datainfo_file = "all_metadata.json"
    
    basename = os.path.basename(pth_path)
    args.model = basename.split('-')[1].split('.')[0]
    task = basename.split('-')[0]
    args.use_tasks = f"['REC', 'LNM', 'TD', 'TI', 'CE', 'PI']"
    
    # fixed seed
    seed = 17
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # dataset 
    mean_std = ([175.14728804175988, 110.57123792228117, 176.73598615775617], [21.239463551725915, 39.15991384752335, 10.99100631656543])
    train_loader, val_loader, test_loader = GetDataLoader(0, mean_std, args, True)
        
    
    # running setting
    model = GetModel(args).cuda()
    assert pth_path is not None, "load_path can't be None."
    print(f"Load from {pth_path}!!!", flush=True)
    cp = load_model(pth_path)
    
    flag = False
    for k, v in cp['model'].items():
        if 'fusion_block' in k:
            flag = True
            break
    
    if flag:
        pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
        pretrain = {k: v for k, v in pretrain.items() if k in model.state_dict()}
        model.load_state_dict(pretrain)
    else:
        model = GetOldModel(args).cuda()
        pretrain = {k.replace('module.', ''): v for k, v in cp['model'].items()}
        pretrain = {k: v for k, v in pretrain.items() if k in model.state_dict()}
        model.load_state_dict(pretrain)
    
    model = MultiTaskModel(model).cuda()
    
    # CAM Methods
    args.method = 'gradcam++'
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,  # unused
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}


    try:  # for vit_small_pathology, vit_base_imagenet
        target_layers = [
            model.model.backbone.extractor.blocks[-2].norm1,
            model.model.backbone.extractor.blocks[-1].norm1,   
        ]
    except:
        target_layers = [
            model.model.backbone.extractor.trunk.blocks[-2].norm1,
            model.model.backbone.extractor.trunk.blocks[-1].norm1
        ]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")


    # current on Baseline directory
    labels_counter = defaultdict(int)
    Analysis_Name = f"{task}_{args.method}"
    os.makedirs(f'./Data/{Analysis_Name}_Results', exist_ok=True)
    # loader = train_loader
    loader = itertools.chain(val_loader, test_loader)
    acc, err = 0, 0
    for x, y, ids in loader:
        
        if pid_list is not None and int(ids[0].item()) not in pid_list:
            continue
        
        label = []
        x = x.cuda()
        for k, v in y.items():
            y[k] = v.cuda()
            label.append(v.item())
        out_y = y
        label = tuple(label)
        
        
        model.eval()
        with torch.no_grad():
            output = model(x)[0]
            pred = []
            for k, v in output.items():
                temp = nn.Softmax(dim=1)(v)
                temp = temp.cpu().data.numpy()
                v = temp.argmax(axis=1)[0]
                pred.append(v)
            pred = tuple(pred)
            
        if pred != label:
            err += 1
            print(f"Error: {ids[0].item()} pred: {pred} label: {label}")
        else:
            acc += 1
            print(f"Correct: {ids[0].item()} pred: {pred} label: {label}")
        
        if labels_counter[label] > 3:
            print(f"Skip: {ids[0].item()} pred: {pred} label: {label} Due to enough samples.")
            continue
        labels_counter[label] += 1
        
        ids = ids.cpu().data.numpy() 
        loss_function = [lambda x: model.loss(x, out_y)]
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        scale_cam = cam(input_tensor=x, targets=loss_function, eigen_smooth=True)
        
        # visualization
        for i, pid in enumerate(ids):
            print(f"Processing {pid} ...", flush=True)
            os.makedirs(f'./Data/{Analysis_Name}_Results/label={label}_{pid}', exist_ok=True)
            
            pid_dir = glob.glob(f"{origin_images_path}/*{pid}*")
            image_names = ['01_2X', "01_4X", "01_10X", "02_2X", "02_4X", "02_10X"]
            
            for j, name in enumerate(image_names):
                image = cv2.imread(f"{pid_dir[0]}/{name}.jpg")
                rgb_img_uint8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_img = np.float32(rgb_img_uint8) / 255
                
                grayscale_cam = scale_cam[i * 6 + j, :]
                grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

                cam_image = show_cam_on_image(rgb_img, grayscale_cam)
                cv2.imwrite(f'./Data/{Analysis_Name}_Results/label={label}_{pid}/{j}th_score.jpg', cam_image)
                cv2.imwrite(f'./Data/{Analysis_Name}_Results/label={label}_{pid}/{j}th_rgb.jpg', rgb_img_uint8)
            
        # del torch tensor
        del x, y, ids, scale_cam, cam, loss_function
        torch.cuda.empty_cache()
        
    print(f"Accuracy: {acc / (acc + err)}")  # 0.155


if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    gpu_id = "1"

    origin_images_path = "/home/Guanjq/HuangData/Multi-OSCCPI-Dataset/"
    pid_list = None
    pth_path = "./BestCheckpoints/Multitask-vit_small_p16_pathology-reinhard.pth"
   
    # inference setting
    os.chdir(os.path.dirname(__file__) + "/../")
    
    work(pth_path, pid_list)
    
    