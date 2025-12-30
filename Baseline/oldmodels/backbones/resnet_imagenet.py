import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torchvision
     
class ResNetImagenet(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args, layers):
        super(ResNetImagenet, self).__init__()
        
        self.img_size = args.img_size
        if layers == 18:
            self.extractor = torchvision.models.resnet18(pretrained=True)
            embed_dim = 512
        elif layers == 34:
            self.extractor = torchvision.models.resnet34(pretrained=True)
            embed_dim = 512
        elif layers == 50:
            self.extractor = torchvision.models.resnet50(pretrained=True)
            embed_dim = 2048
        elif layers == 101:
            self.extractor = torchvision.models.resnet101(pretrained=True)
            embed_dim = 2048
            
        self.extractor.fc = nn.Identity()
        params_sum = sum([param.nelement() for param in self.extractor.parameters()])
        print(f"extractor size: {params_sum / 1e6}Mb")
        
        self.neck = nn.Sequential(
            nn.Linear(embed_dim * self.ensemble_num, 1024),  # 0
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
        x = torch.reshape(x, (-1, 2048 * self.ensemble_num))
        return self.neck(x)
    