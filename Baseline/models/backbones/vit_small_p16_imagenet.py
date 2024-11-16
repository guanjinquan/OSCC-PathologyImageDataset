import torch.nn as nn
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源
from timm.models.vision_transformer import vit_small_patch16_224, resize_pos_embed, VisionTransformer
# from torchvision.models import vit_b_16   # can't adjust model into 512x512 size input
import torch


def get_vit(args):
    # vit_Small_patch32_224
    pretrain_model = vit_small_patch16_224(pretrained=True).state_dict()
    
    model = VisionTransformer(  # 512 / 16 = 32 -> 32 * 32 patches
        img_size=args.img_size, patch_size=16, embed_dim=384, depth=12, num_heads=12
    )
    
    pretrain_model['pos_embed'] = resize_pos_embed(pretrain_model['pos_embed'], model.pos_embed)
    model.load_state_dict(pretrain_model)
    
    model.head = nn.Identity()
    return model


class ViTSmallImagenet(nn.Module):
    
    def __init__(self, args):
        super(ViTSmallImagenet, self).__init__()
        self.extractor = get_vit(args)
         
    def forward(self, x):
        x = self.extractor(x)
        return x


def get_vit_small_imagenet(args):
    model = ViTSmallImagenet(args)
    return model, 384


if __name__ == '__main__':
    # print(vit_Small_patch16_224(pretrained=True))
    class Args:
        img_size = 512
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = ViTSmallImagenet(Args()).cuda()
    print(model)
    print(model(torch.randn(1, 3, 512, 512).cuda()).shape)