import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, vit_small_patch16_224
from timm.layers import PatchEmbed
import numpy as np
from scipy import ndimage

def get_vit(args):
    # vit_base_patch32_224
    pretrain_model = vit_small_patch16_224(pretrained=True).state_dict()
    
    model = VisionTransformer(  # 512 / 16 = 32 -> 32 * 32 patches
        img_size=args.img_size, patch_size=16, embed_dim=384, num_heads=6, num_classes=2
    )
    
    # 加载其他权重
    net_state = model.state_dict()
    net_state.update({k:v for k,v in pretrain_model.items() if k in net_state and ('pos_embed' not in k) and ('head' not in k)})
    model.load_state_dict(net_state, strict=False)
    
    # patch_embed 里面就是一个 conv + norm
    posemb = pretrain_model['pos_embed'].detach().numpy()
    posemb_new = model.state_dict()["pos_embed"]
    ntok_new = posemb_new.shape[1]
    posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.shape[1], 1), order=1)
    posemb_zoom = np.expand_dims(posemb_zoom, 0)
    model.pos_embed = nn.Parameter(torch.from_numpy(posemb_zoom))
    
    model.head = nn.Identity()
    return model

class VitImagenet(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(VitImagenet, self).__init__()
        
        self.img_size = args.img_size
        self.extractor = get_vit(args) 
    
        self.neck = nn.Sequential(
            nn.Linear(384 * self.ensemble_num, 768),  # 0
            nn.LayerNorm(768), 
            nn.ReLU(),
        )
        
    def get_backbone_params(self):
        return list(self.extractor.parameters())

    def get_others_params(self):
        backbone_params = set(self.extractor.parameters())
        return [p for p in self.parameters() if p not in backbone_params]

    def forward(self, x):
        assert x.shape[0] % self.ensemble_num == 0
        x = self.extractor(x)
        x = torch.reshape(x, (-1, 384 * self.ensemble_num))
        return self.neck(x)
    
if __name__ == "__main__":
    model = vit_small_patch16_224(pretrained=True)
    print(model)