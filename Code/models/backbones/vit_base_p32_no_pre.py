import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


def get_vit(args):
    model = VisionTransformer(  # 512 / 32 = 16 -> 16 * 16 patches
        img_size=args.simg_ize, patch_size=32, embed_dim=768, num_heads=6, num_classes=2
    )
    
    model.head = nn.Identity()
    return model

class VanillaViT(nn.Module):
    ensemble_num = 6
    
    def __init__(self):
        super(VanillaViT, self).__init__()
        
        self.hidden_dim = 768
        self.img_size = 512
        self.extractor = get_vit() 
    
        self.neck = nn.Sequential(
            nn.Linear(768 * self.ensemble_num, 768),  # 0
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
        x = torch.reshape(x, (-1, 768 * self.ensemble_num))
        return self.neck(x)