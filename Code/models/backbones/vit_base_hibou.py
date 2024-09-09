from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F

def get_hibou(args):
    hibou_path = "/mnt/home/Guanjq/BackupWork/PathoCls/Code/models/backbones/pretrained_weight/hibou-b"
    # processor = AutoImageProcessor.from_pretrained(hibou_path)
    model = AutoModel.from_pretrained(hibou_path)
    
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
    ensemble_num = 6
    
    def __init__(self, args):
        super(VitHibou, self).__init__()
        
        self.extractor, self.patch_size = get_hibou(args)

        self.neck = nn.Sequential(
            nn.Linear(768 * self.ensemble_num, 768),  # 0
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
        if x.shape[-1] % self.patch_size != 0:
            # 使用 pad 0 的方式中心补全图片
            pad = (self.patch_size - x.shape[-1] % self.patch_size) // 2
            x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        x = self.extractor(x).last_hidden_state[:, 0, :]
        x = torch.reshape(x, (-1, 768 * self.ensemble_num))
        feat = self.neck(x)
        return feat



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hibou-b-p14')
    args = parser.parse_args()
    model = VitHibou(args).cuda()
    input = torch.randn(6, 3, 512, 512).cuda()
    print(model(input).shape)
