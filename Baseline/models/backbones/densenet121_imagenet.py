import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源
from torchvision.models.densenet import densenet121


class Densenet121Imagenet(nn.Module):
    
    def __init__(self, args):
        super(Densenet121Imagenet, self).__init__()
        
        self.extractor = densenet121(pretrained=True)
        self.extractor.classifier = nn.Identity()
        
        params_sum = sum([param.nelement() for param in self.extractor.parameters()])
         
    def get_backbone_params(self):
        return list(self.extractor.parameters())

    def get_others_params(self):
        backbones = set(self.get_backbone_params())
        return [p for p in self.parameters() if p not in backbones]
        
    def forward(self, x):
        x = self.extractor(x)
        return x


def get_densenet121_imagenet(args):
    model = Densenet121Imagenet(args)
    return model, 1024


if __name__ == "__main__":
    model = Densenet121Imagenet(None)
    print("Params : ", sum([param.nelement() for param in model.parameters()]) / (1024 * 1024) * 4, "MB")