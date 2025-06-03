import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torchvision
     
class ResNetImagenet(nn.Module):

    def __init__(self, args, layers):
        super(ResNetImagenet, self).__init__()
        self.freezed_backbone = args.freezed_backbone
        self.img_size = args.img_size
        if layers == 18:
            self.extractor = torchvision.models.resnet18(pretrained=True)
            self.embed_dim = 512
        elif layers == 34:
            self.extractor = torchvision.models.resnet34(pretrained=True)
            self.embed_dim = 512
        elif layers == 50:
            self.extractor = torchvision.models.resnet50(pretrained=True)
            self.embed_dim = 2048
        elif layers == 101:
            self.extractor = torchvision.models.resnet101(pretrained=True)
            self.embed_dim = 2048
            
        self.extractor.fc = nn.Identity()
        params_sum = sum([param.nelement() for param in self.extractor.parameters()])


    def get_backbone_params(self):
        return list(self.extractor.parameters())

    def get_others_params(self):
        backbones = set(self.get_backbone_params())
        return [p for p in self.parameters() if p not in backbones]
        
    def forward(self, x):
        if self.freezed_backbone:
            with torch.no_grad():
                x = self.extractor(x)
        else:
            x = self.extractor(x)
        return x


def get_resnet_imagenet(args, layers):
    model = ResNetImagenet(args, layers)
    return model, model.embed_dim

if __name__ == "__main__":
    class Args:
        img_size = 512
    model = get_resnet_imagenet(Args(), 50)[0]
    # print(model)
    img = torch.zeros((3, 2592, 1944))
    out = model(img.unsqueeze(0))
    print("Output shape: ", out.shape)
    # print("Params : ", sum([param.nelement() for param in model.parameters()]) / (1024 * 1024) * 4, "MB")