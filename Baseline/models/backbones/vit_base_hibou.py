# from models.backbones.Hibou_modules.modeling_dinov2 import Dinov2ModelWithRegisters
# from models.backbones.Hibou_modules.configuration_dinov2 import Dinov2Config
from transformers import AutoModel
import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F


def get_hibou(args):
    hibou_path = "./Baseline/models/backbones/pretrained_weight/hibou-b/"
    model = AutoModel.from_pretrained(hibou_path, trust_remote_code=True)
    
    if 'p14' in args.model:
        patch_size = 14
    elif 'p16' in args.model:
        patch_size = 16
        model.config.patch_size = 16
        # 使用双线性插值加载权重
        new_embed = torch.nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
        old_embed = model.embeddings.patch_embeddings.projection
        new_embed.weight.data = F.interpolate(old_embed.weight.data, size=(16, 16), mode='bicubic', antialias=True)
        new_embed.bias.data = old_embed.bias.data.clone()
        model.embeddings.patch_embeddings.projection = new_embed
    
    return model, patch_size


class VitHibou(nn.Module):
    def __init__(self, args):
        super(VitHibou, self).__init__()
        self.extractor, self.patch_size = get_hibou(args)

    def forward(self, x):
        assert x.shape[0] % self.ensemble_num == 0
        if x.shape[-1] % self.patch_size != 0:
            # 使用 pad 0 的方式中心补全图片
            pad = (self.patch_size - x.shape[-1] % self.patch_size) // 2
            x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        x = self.extractor(x).last_hidden_state[:, 0, :]
        return x


def get_vit_base_hibou(args):
    model = VitHibou(args)
    return model, 768


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..', ".."))
    class Args:
        model = "vit_base_hibou_p16"
        img_size = 512
    model = get_vit_base_hibou(Args())[0]
    print("Params : ", sum([param.nelement() for param in model.parameters()]) / (1024 * 1024) * 4, "MB")