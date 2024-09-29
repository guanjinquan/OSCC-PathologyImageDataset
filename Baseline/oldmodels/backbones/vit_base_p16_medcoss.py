import torch
import torch.nn as nn
from models.backbones.MedCoSS_modules.Uni_model import Unified_Model
from torchvision.models.vision_transformer import MLPBlock

class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, attn_mask):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, attn_mask=attn_mask, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y    

# ##################################################################

def get_vit(args):
    model = Unified_Model(now_2D_input_size=args.img_size)
    checkpoint = torch.load("./Baseline/models/backbones/pretrained_weight/medcoss-epoch299.pth")
    # 预训练有一些decoder权重没有用到，需要筛掉
    checkpoint = {k: v for k, v in checkpoint['model'].items() if k in model.state_dict().keys()}
    model.load_state_dict(checkpoint)
    model._change_input_chans_2D(3)
    return model
    

class ViTBaseMedCoSS(nn.Module):
    ensemble_num = 6
    
    def __init__(self, args):
        super(ViTBaseMedCoSS, self).__init__()
        self.img_size = args.img_size
        self.hidden_dim = 768
        self.extractor = get_vit(args) 
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
        x = self.extractor({"modality": "2D image", "data": x}) 
        x = torch.reshape(x, (-1, 768 * self.ensemble_num))
        feat = self.neck(x)
        return feat

