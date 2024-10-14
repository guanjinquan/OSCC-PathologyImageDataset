import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源
from torchvision.models import vit_b_16
import torch

class ViTBaseImagenet(nn.Module):
    
    def __init__(self, args):
        super(ViTBaseImagenet, self).__init__()
        
        self.extractor = vit_b_16(pretrained=True)
        self.extractor.heads = nn.Identity()
        self.extractor.classifier = nn.Identity()
        
        params_sum = sum([param.nelement() for param in self.extractor.parameters()])
        print(f"extractor size: {params_sum / 1e6}Mb")
         
    def get_backbone_params(self):
        return list(self.extractor.parameters())

    def get_others_params(self):
        backbones = set(self.get_backbone_params())
        return [p for p in self.parameters() if p not in backbones]
        
    def forward(self, x):
        x = self.extractor(x)
        return x


def get_vit_base_imagenet(args):
    model = ViTBaseImagenet(args)
    return model, 768


if __name__ == '__main__':
    model = ViTBaseImagenet(None).cuda()
    print(model)
    print(model(torch.randn(1, 3, 224, 224).cuda()).shape)