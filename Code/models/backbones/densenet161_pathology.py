import safetensors.torch
import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源
import timm
import torch
from safetensors import safe_open
from torchvision.models.densenet import densenet161


def get_densenet161(args):
    pretrained_path = "/mnt/home/Guanjq/BackupWork/PathoCls/Code/models/backbones/DenseNet161_pathology/model.safetensors"
    model = densenet161(pretrained=False)
    pretrain = {}
    with safe_open(pretrained_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            if 'classifier' not in k:
                pretrain[k] = f.get_tensor(k)
    model.load_state_dict(pretrain, strict=False)  # don't load classifier
    return model



class Densenet161Pathology(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(Densenet161Pathology, self).__init__()
        
        self.extractor = get_densenet161(args)
        self.extractor.classifier = nn.Linear(2208, 768)
        
        params_sum = sum([param.nelement() for param in self.extractor.parameters()])
        print(f"extractor size: {params_sum / 1e6}Mb")

        self.neck = nn.Sequential(
            nn.Linear(768 * self.ensemble_num, 1024),  # 0
            nn.LayerNorm(1024), 
            nn.ReLU(),
            
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
        x = torch.reshape(x, (-1, 768 * self.ensemble_num))
        return self.neck(x)

if __name__ == "__main__":
    model = Densenet161Pathology(None).cuda()
    out = model(torch.randn(6, 3, 224, 224).cuda())
    print(out.shape)