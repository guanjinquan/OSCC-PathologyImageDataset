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


class TaskSpecificModel(nn.Module):
    def __init__(self, task, model):
        super(TaskSpecificModel, self).__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        x = self.model(x)
        return x[self.task]


if __name__ == '__main__':
    
    # setting config
    origin_images_path = "/home/Guanjq/HuangData/PathologyImages/"
    gpu_id = "1"
    load_pth_path = "./BestCheckpoints/TI-vit_small_p16_pathology.pth"
    
    # inference setting
    os.chdir(os.path.dirname(__file__) + "/../")
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    args.split_filename = "split_seed=2024.json"
    args.datainfo_file = "all_metadata.json"
    
    basename = os.path.basename(load_pth_path)
    args.model = basename.split('-')[1].split('.')[0]
    task = basename.split('-')[0]
    
    # fixed seed
    seed = int(args.seed)
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
    assert load_pth_path is not None, "load_path can't be None."
    print(f"Load from {load_pth_path}!!!", flush=True)
    cp = load_model(load_pth_path)
    
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
    
    model = TaskSpecificModel(task, model).cuda()
    

    # CAM Methods
    args.method = 'scorecam'
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


    # target_layers = [model.model.backbone.extractor.blocks[-1].norm1]
    target_layers = [model.model.backbone.extractor.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")


    # current on Baseline directory
    Analysis_Name = f"{args.runs_id}_{args.method}"
    os.makedirs(f'./Data/{Analysis_Name}_GradCAM_Results', exist_ok=True)
    for x, y, ids in train_loader:
        x = x.cuda()
        t = torch.ones((1)).long().cuda()
        for k, v in y.items():
            y[k] = v.cuda()
        ids = ids.cpu().data.numpy() 
        loss_function = [lambda x: nn.CrossEntropyLoss()(x.reshape(1, -1), t)]
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        scale_cam = cam(input_tensor=x, targets=loss_function, eigen_smooth=True)
        
        # visualization
        for i, pid in enumerate(ids):
            print(f"Processing {pid}...", flush=True)
            os.makedirs(f'./Data/{Analysis_Name}_GradCAM_Results/{pid}', exist_ok=True)
            
            pid_dir = glob.glob(f"{origin_images_path}/*{pid}*")
            image_names = ['01_2X', "01_4X", "01_10X", "02_2X", "02_4X", "02_10X"]
            
            for j, name in enumerate(image_names):
                image = cv2.imread(f"{pid_dir[0]}/{name}.jpg")
                rgb_img_uint8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_img = np.float32(rgb_img_uint8) / 255
                grayscale_cam = scale_cam[i * 6 + j, :]
                grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[2]))
                cam_image = show_cam_on_image(rgb_img, grayscale_cam)
                grayscale_cam_uint8 = np.uint8(grayscale_cam * 255)
                # cv2.imwrite(f'./Data/{Analysis_Name}_GradCAM_Results/{pid}/{args.method}_cam_{pid}_{j}th_label={y["REC"][i].item()}.jpg', cam_image)
                grayscale_cam_uint8 = cv2.applyColorMap(grayscale_cam_uint8, cv2.COLORMAP_JET)
                cv2.imwrite(f'./Data/{Analysis_Name}_GradCAM_Results/{pid}/{args.method}_cam_{pid}_{j}th_label={y["REC"][i].item()}_heatmap.jpg', grayscale_cam_uint8)
                cv2.imwrite(f'./Data/{Analysis_Name}_GradCAM_Results/{pid}/{args.method}_cam_{pid}_{j}th_label={y["REC"][i].item()}_rgb.jpg', rgb_img_uint8)
                exit(0)
            
        del x, y, ids, scale_cam
        torch.cuda.empty_cache()
        