from torch import nn
import torch
from torchvision.models.vision_transformer import EncoderBlock


# concat
class ConcatBlock(nn.Module):
    def __init__(self, num_feat, in_dim, out_dim) -> None:
        super(ConcatBlock, self).__init__()
        
        self.num_feat = num_feat
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.neck = nn.Sequential(
            nn.Linear(self.in_dim * self.num_feat, self.out_dim),
            nn.LayerNorm(self.out_dim), 
            nn.ReLU(),
        )
        print("Params : ", sum(p.numel() for p in self.parameters()) / (1024 * 1024) * 4, " MB")

    def forward(self, x):
        # print("INPUT Shape = ", x.shape)
        # x.shape = (batch_size * num_feat, in_dim)
        assert x.shape[1] == self.in_dim, f"x.shape[1]: {x.shape[1]}, self.in_dim: {self.in_dim}"
        assert x.shape[0] % self.num_feat == 0, f"x.shape[0]: {x.shape[0]}, self.num_feat: {self.num_feat}"
        x = torch.reshape(x, (-1, self.in_dim * self.num_feat))
        # print("FUSE Shape = ", x.shape)
        return self.neck(x)
    
    
# Low-Rank Multimodal TensorFusion
# ref: https://github.com/jacquelinelala/GFN/blob/master/networks/GFN_4x.py
class LowRankFusionBlock(nn.Module):
    def __init__(self, num_feat, in_dim, out_dim) -> None:
        super(LowRankFusionBlock, self).__init__()
        
        self.rank = 16
        
        self.num_feat = num_feat
        self.in_dim = in_dim
        self.out_dim = out_dim
 
        self.factor = nn.Parameter(
            # r, d+1, d+1
            torch.randn(self.rank, self.out_dim+1, self.out_dim+1),
            requires_grad=True
        ) 
        
        self.transition = nn.Linear(self.in_dim, self.out_dim)
        
        self.neck = nn.Sequential(
            nn.Linear(in_features=self.rank*(self.out_dim+1), out_features=self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),  
        )
        print("Params : ", sum(p.numel() for p in self.parameters()) / (1024 * 1024) * 4, " MB")
    
    def forward(self, x):
        device = x.device
        b, n = x.shape[0] // self.num_feat, self.num_feat
        
        # [b * n, in_dim] -> [n, b, in_dim]
        x = x.view(b, n, self.in_dim).permute(1, 0, 2)
        
        # [n, b, in_dim] -> [n, b, out_dim]
        x = self.transition(x)
        
        # [n, b, out_dim] -> [n, b, out_dim+1]
        x = torch.cat([torch.ones(n, b, 1).to(device), x], dim=-1)
        
        # [1, 1, r, out_dim+1, out_dim+1] @ [n, b, 1, out_dim+1, 1] -> [n, b, rank, out_dim+1]
        x = (self.factor @ x.unsqueeze(2).unsqueeze(4)).squeeze(4)
        
        # [n, b, rank, out_dim+1] -> [b, rank, out_dim+1]
        x = x.prod(dim=0)
        
        # [b, r * (out_dim+1)] -> [b, out_dim]
        x = x.view(b, -1)
        x = self.neck(x)
        
        return x



# gated-fusion block
# ref: https://github.com/jacquelinelala/GFN/blob/master/networks/GFN_4x.py
# Or another paper: Gated multimodal units for information fusion. https://arxiv.org/pdf/1702.01992
class GatedFusionBLock(nn.Module):
    def __init__(self, num_feat, in_dim, out_dim) -> None:
        super(GatedFusionBLock, self).__init__()
        
        self.num_feat = num_feat
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.transition = nn.Linear(self.in_dim, self.out_dim)
        print("Params : ", sum(p.numel() for p in self.parameters()) / (1024 * 1024) * 4, " MB")
        
    def forward(self, x):
        assert x.shape[1] == self.in_dim, f"x.shape[1]: {x.shape[1]}, self.in_dim: {self.in_dim}"
        assert x.shape[0] % self.num_feat == 0, f"x.shape[0]: {x.shape[0]}, self.num_feat: {self.num_feat}"

        # [b*n, in_d] -> [n, b, out_d]
        x = self.transition(x)
        # [b*n, d] -> [n, b, d]
        x = torch.reshape(x, (-1, self.num_feat, self.out_dim)).permute(1, 0, 2)
        # sigmoid and tanh
        gate = torch.sigmoid(x)
        y = torch.tanh(x)         
        # [n, b, d] -> [b, d]
        y = (gate * y).sum(dim=0)
        return y


# self-attention fusion
class MSAFusionBlock(nn.Module):
    def __init__(self, num_feat, in_dim, out_dim) -> None:
        super(MSAFusionBlock, self).__init__()
        
        self.layers_num = 2
        self.attn_heads = 8
        self.num_feat = num_feat
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.transition = nn.Linear(self.in_dim, self.out_dim)
        
        assert self.out_dim % self.attn_heads == 0, f"out_dim must be a multiple of {self.attn_heads}, but got {self.out_dim}"
        self.Attn_Layers = nn.Sequential(
            *[EncoderBlock(
                num_heads = self.attn_heads,
                hidden_dim = self.out_dim,  
                mlp_dim = self.out_dim * 2,   
                dropout = 0,
                attention_dropout = 0,
            ) for _ in range(self.layers_num)]
        )
        print("Params : ", sum(p.numel() for p in self.parameters()) / (1024 * 1024) * 4, " MB")
        
    def forward(self, x):
        assert x.shape[1] == self.in_dim, f"x.shape[1]: {x.shape[1]}, self.in_dim: {self.in_dim}"
        assert x.shape[0] % self.num_feat == 0, f"x.shape[0]: {x.shape[0]}, self.num_feat: {self.num_feat}"
        
        # Transition [b*n, in_d] -> [b, n, out_d]
        x = self.transition(x).reshape(-1, self.num_feat, self.out_dim)
        
        # Attention [b, n, out_d] -> [b, n, out_d]
        x = self.Attn_Layers(x)
        
        # Mean Pooling
        x = x.mean(dim=1)
        return x
    
    
if __name__ == "__main__":
    # [1 * 6, 384]
    input = torch.randn(6, 384).cuda()
    
    # concat
    print("Working on ConcatBlock")
    model = ConcatBlock(6, 384, 768).cuda()
    out = model(input)
    
    # low-rank fusion
    print("Working on LowRankFusionBlock")
    model = LowRankFusionBlock(6, 384, 768).cuda()
    out = model(input)
    
    # gated-fusion
    print("Working on GatedFusionBLock")
    model = GatedFusionBLock(6, 384, 768).cuda()
    out = model(input)
    
    # self-attention fusion
    print("Working on MSAFusionBlock")
    model = MSAFusionBlock(6, 384, 768).cuda()
    out = model(input)
