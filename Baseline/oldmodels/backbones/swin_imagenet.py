import torch
import torch.nn as nn
from timm.models.swin_transformer_v2 import SwinTransformerV2, swinv2_base_window12to16_192to256
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

def get_swin(args):
    state = swinv2_base_window12to16_192to256(pretrained=True, num_classes=1000).state_dict()
    # swin transformer 可以改变输入图片大小，但是要保证window_size能被整除
    # 找到合适的权重就行
    
    model = SwinTransformerV2(
        img_size=args.img_size,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=16,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False    
    )

    model.load_state_dict({k:v for k, v in state.items() if k in model.state_dict().keys()}, strict=True)
    model.head.fc = nn.Identity()
    model.patch_embed.img_size = None

    return model, 1024


class SwinImageNet(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(SwinImageNet, self).__init__()
        
        self.extractor, self.embed_dim = get_swin(args)
        
        self.neck = nn.Sequential(
            nn.Linear(self.embed_dim * self.ensemble_num, 768),  # 0
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
        x = torch.reshape(x, (-1, self.embed_dim * self.ensemble_num))
        return self.neck(x)


if __name__ == "__main__":
    model = SwinImageNet()
    input = torch.ones((6, 3, 512, 512))
    print(model(input).shape)