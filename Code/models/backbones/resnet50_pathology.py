import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torchvision.models.resnet import Bottleneck, ResNet

class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = kwargs.get("num_classes", None)
        self.fc = torch.nn.Linear(2048, self.hidden_dim, bias=True)

    def forward(self, x): 
        x = self.conv1(x)  # down sample 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # down sample 2

        x = self.layer1(x)
        x = self.layer2(x)  # down sample 2
        x = self.layer3(x)  # down sample 2
        x = self.layer4(x)  # down sample 2
        
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet_get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    torch.nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
    torch.nn.init.zeros_(model.fc.bias)
    if pretrained:
        pretrained_url = resnet_get_pretrained_url(key)
        state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        net_dict = {k:v for k, v in state_dict.items() if k in model.state_dict().keys() and k != "fc"}
        verbose = model.load_state_dict(net_dict, strict=False)
        print(verbose)  # _IncompatibleKeys(missing_keys=['pos_embed', 'head.weight', 'head.bias'], unexpected_keys=[])
    return model

class ResNetPathology(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(ResNetPathology, self).__init__()
        
        self.img_size = args.img_size
        self.extractor = resnet50(pretrained=True, progress=True, key="BT", num_classes=1024)
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

if __name__ == "__main__":
    model = ResNetPathology()
    x = torch.randn(6, 3, 224, 224)
    y = model(x)
    print(y.shape)