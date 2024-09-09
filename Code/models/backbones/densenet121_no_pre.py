import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源
import timm
import torch
from torchvision.models.densenet import densenet121

class Densenet121_no_pre(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(Densenet121_no_pre, self).__init__()
        
        self.extractor = densenet121()
        self.extractor.classifier = nn.Identity()
        
        params_sum = sum([param.nelement() for param in self.extractor.parameters()])
        print(f"extractor size: {params_sum / 1e6}Mb")

        self.neck = nn.Sequential(
            nn.Linear(1024 * self.ensemble_num, 1024),  # 0
            nn.LayerNorm(1024), 
            nn.ReLU(),
            # nn.Dropout(0),
            
            nn.Linear(1024, 768),  # 4
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
        return self.neck(x)
    