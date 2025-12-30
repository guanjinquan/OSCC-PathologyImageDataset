import timm
import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch


def get_Uni():
    Uni_path = "./Baseline/models/backbones/pretrained_weight/UNI"
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(Uni_path, "pytorch_model.bin"), map_location="cpu"), strict=True)
    return model


class VitUni(nn.Module):
    def __init__(self, args):
        super(VitUni, self).__init__()
        self.img_size = args.img_size
        self.extractor = get_Uni()
    
    def forward(self, x):
        x = self.extractor(x)
        return x


def get_vit_base_uni(args):
    model = VitUni(args)
    return model, 1024