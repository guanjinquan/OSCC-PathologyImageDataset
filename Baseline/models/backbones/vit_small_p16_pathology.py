import torch.nn as nn
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from scipy import ndimage
from timm.models.vision_transformer import VisionTransformer
import torch


def vit_get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(args, pretrained, progress, key):
    patch_size = 16
    img_size = args.img_size
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=384, num_heads=6)
    
    model.head = nn.Identity()

    if pretrained:
        pretrained_url = vit_get_pretrained_url(key)
        state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        
        # 获取除了pos_embeds和head的参数
        net_dict = {k:v for k, v in state_dict.items() if k in model.state_dict().keys() and k != "pos_embed"}
        
        # 获取pos_embeds
        posemb = state_dict["pos_embed"]
        posemb_new = model.state_dict()["pos_embed"]
        ntok_new = posemb_new.size(1)
        posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
        posemb_zoom = np.expand_dims(posemb_zoom, 0)
        net_dict.update({"pos_embed": torch.from_numpy(posemb_zoom)})
        
        # 更新参数
        verbose = model.load_state_dict(net_dict, strict=False)
        print(verbose)  # _IncompatibleKeys(missing_keys=['head.weight', 'head.bias'], unexpected_keys=[])
        
    return model


class VitPathology(nn.Module):
    def __init__(self, args):
        super(VitPathology, self).__init__()
        self.freezed = args.freezed_backbone
        self.extractor = vit_small(args, pretrained=True, progress=True, key="DINO_p16")
        
    def forward(self, x):
        if self.freezed:
            with torch.no_grad():
                x = self.extractor(x)
        else:
            x = self.extractor(x)
        return x


def get_vit_base_pathology(args):
    model = VitPathology(args)
    return model, 384

if __name__ == "__main__":
    class Args:
        # img_size = 512
        img_size = (1944, 2592)
        freezed_bakcbone = True
    model = get_vit_base_pathology(Args())[0].cuda(2)
    print("Params : ", sum([param.nelement() for param in model.parameters()]) / (1024 * 1024) * 4, "MB")
    input = torch.randn(1, 3, 1944, 2592).cuda(2)
    output = model(input)
    print("Output shape: ", output.shape)