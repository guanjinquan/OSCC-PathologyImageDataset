import timm
import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F


def get_Uni():
    Uni_path = "./Baseline/models/backbones/pretrained_weight/UNI"
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(Uni_path, "pytorch_model.bin"), map_location="cpu"), strict=True)
    return model


class VitUni(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(VitUni, self).__init__()
        self.img_size = args.img_size
        self.extractor = get_Uni()
        self.neck = nn.Sequential(
            nn.Linear(1024 * self.ensemble_num, 768),  # 0
            nn.LayerNorm(768), 
            nn.ReLU(),
        )
    
    def get_backbone_params(self):
        return list(self.extractor.parameters())

    def get_others_params(self):
        backbones = set(self.get_backbone_params())
        return [p for p in self.parameters() if p not in backbones]
        
    def forward(self, x):
        assert x.shape[0] % self.ensemble_num == 0
        x = self.extractor(x)
        x = torch.reshape(x, (-1, 1024 * self.ensemble_num))
        feat = self.neck(x)
        return feat

