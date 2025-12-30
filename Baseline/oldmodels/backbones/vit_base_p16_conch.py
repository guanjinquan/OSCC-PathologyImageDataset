import sys
from ExternalLibs.CONCH.conch.open_clip_custom import create_model_from_pretrained
import torch.nn as nn
import torch


def get_Conch(args):
    Conch_path = "./Baseline/models/backbones/pretrained_weight/CONCH/pytorch_model.bin"
    model = create_model_from_pretrained("conch_ViT-B-16", Conch_path)[0]  # (model, transform) 
    model = model.visual
    return model


class VitConch(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(VitConch, self).__init__()
        
        self.img_size = args.img_size
        self.extractor = get_Conch(args)
        self.neck = nn.Sequential(
            nn.Linear(512 * self.ensemble_num, 768),  # 0
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
        # 加载的权重output的dim=512
        x = self.extractor(x)[0]  # (pooled_ouput, tokens_output).shape = (b, 512), (b, n, 768)
        x = torch.reshape(x, (-1, 512 * self.ensemble_num))
        feat = self.neck(x)
        return feat
