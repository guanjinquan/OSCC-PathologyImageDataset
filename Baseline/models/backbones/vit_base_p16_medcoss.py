import torch
import torch.nn as nn
from models.backbones.MedCoSS_modules.Uni_model import Unified_Model


def get_vit(args):
    model = Unified_Model(now_2D_input_size=args.img_size)
    checkpoint = torch.load("./Baseline/models/backbones/pretrained_weight/medcoss-epoch299.pth")
    # 预训练有一些decoder权重没有用到，需要筛掉
    checkpoint = {k: v for k, v in checkpoint['model'].items() if k in model.state_dict().keys()}
    model.load_state_dict(checkpoint)
    model._change_input_chans_2D(3)
    return model
    

class ViTBaseMedCoSS(nn.Module): 
    def __init__(self, args):
        super(ViTBaseMedCoSS, self).__init__()
        self.img_size = args.img_size
        self.hidden_dim = 768
        self.extractor = get_vit(args) 

    def forward(self, x):
        x = self.extractor({"modality": "2D image", "data": x}) 
        return x


def get_vit_base_medcoss(args):
    model = ViTBaseMedCoSS(args)
    return model, model.hidden_dim