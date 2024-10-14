from models.backbones import *
from models.tasks import MultiTaskModel, metrics
from models.fusion_blocks import *

def GetModel(args):

    if args.model == 'resnet50_imagenet':
        backbone, embed_dim = get_resnet_imagenet(args, 50)
    elif args.model == 'densenet121_imagenet':
        backbone, embed_dim = get_densenet121_imagenet(args)
    elif args.model == 'vit_small_p16_pathology':
        backbone, embed_dim = get_vit_base_pathology(args)
    elif args.model == 'swin_imagenet':
        backbone, embed_dim = get_swin_imageNet(args)
    elif args.model == 'vit_base_p16_medcoss':
        backbone, embed_dim = get_vit_base_medcoss(args)
    elif args.model == 'vit_base_p14_hibou' or args.model == 'vit_base_p16_hibou':
        backbone, embed_dim = get_vit_base_hibou(args)
    elif args.model == 'vit_base_p16_uni':
        backbone, embed_dim = get_vit_base_uni(args)
    elif args.model == 'vit_base_p16_conch':
        backbone, embed_dim = get_vit_base_conch(args)
    elif args.model == 'vit_base_imagenet':
        backbone, embed_dim = get_vit_base_imagenet(args)
    else:
        raise ValueError("model not supported")
    
    num_feat = 6 if args.data_type == 'ALL' else 3
    hidden_dim = 768
    if args.fusion_block == 'concat':
        fusion_block = ConcatBlock(num_feat, embed_dim, hidden_dim)
    elif args.fusion_block == 'LMF':
        fusion_block = LowRankFusionBlock(num_feat, embed_dim, hidden_dim)
    elif args.fusion_block == 'gated':
        fusion_block = GatedFusionBLock(num_feat, embed_dim, hidden_dim)
    elif args.fusion_block == 'transformer_encoder':
        fusion_block = MSAFusionBlock(num_feat, embed_dim, hidden_dim)
    else:
        raise ValueError("fusion block not supported")
    
    model = MultiTaskModel(backbone, fusion_block, hidden_dim, args)
    return model 


def GetOldModel(args):

    if args.model == 'resnet50_imagenet':
        backbone, embed_dim = get_resnet_imagenet(args, 50)
    elif args.model == 'densenet121_imagenet':
        backbone, embed_dim = get_densenet121_imagenet(args)
    elif args.model == 'vit_small_p16_pathology':
        backbone, embed_dim = get_vit_base_pathology(args)
    elif args.model == 'swin_imagenet':
        backbone, embed_dim = get_swin_imageNet(args)
    elif args.model == 'vit_base_p16_medcoss':
        backbone, embed_dim = get_vit_base_medcoss(args)
    elif args.model == 'vit_base_p14_hibou' or args.model == 'vit_base_p16_hibou':
        backbone, embed_dim = get_vit_base_hibou(args)
    elif args.model == 'vit_base_p16_uni':
        backbone, embed_dim = get_vit_base_uni(args)
    elif args.model == 'vit_base_p16_conch':
        backbone, embed_dim = get_vit_base_conch(args)
    else:
        raise ValueError("model not supported")
    
    num_feat = 6 if args.data_type == 'ALL' else 3
    hidden_dim = 768
    if args.fusion_block == 'concat':
        fusion_block = ConcatBlock(num_feat, embed_dim, hidden_dim)
    elif args.fusion_block == 'LMF':
        fusion_block = LowRankFusionBlock(num_feat, embed_dim, hidden_dim)
    elif args.fusion_block == 'gated':
        fusion_block = GatedFusionBLock(num_feat, embed_dim, hidden_dim)
    elif args.fusion_block == 'transformer_encoder':
        fusion_block = MSAFusionBlock(num_feat, embed_dim, hidden_dim)
    else:
        raise ValueError("fusion block not supported")
    
    model = MultiTaskModel(backbone, fusion_block, hidden_dim, args)
    return model 