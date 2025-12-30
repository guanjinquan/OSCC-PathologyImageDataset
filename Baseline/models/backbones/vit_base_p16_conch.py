import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', ".."))
from ExternalLibs.CONCH.conch.open_clip_custom import create_model_from_pretrained
import torch.nn as nn

def get_Conch(args):
    Conch_path = "./Baseline/models/backbones/pretrained_weight/CONCH/pytorch_model.bin"
    model = create_model_from_pretrained("conch_ViT-B-16", Conch_path)[0]  # (model, transform) 
    model = model.visual
    return model


class VitConch(nn.Module):
    def __init__(self, args):
        super(VitConch, self).__init__()
        self.img_size = args.img_size
        self.extractor = get_Conch(args)
    
    def forward(self, x):
        # output_dim = 512
        x = self.extractor(x)[0]  # (pooled_ouput, tokens_output).shape = (b, 512), (b, n, 768)
        return x


def get_vit_base_conch(args):
    model = VitConch(args)
    return model, 512


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..', ".."))
    class Args:
        img_size = 512
    model = get_vit_base_conch(Args())[0]
    print("Params : ", sum([param.nelement() for param in model.parameters()]) / (1024 * 1024) * 4, "MB")