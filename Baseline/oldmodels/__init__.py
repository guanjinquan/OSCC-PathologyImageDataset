from oldmodels.backbones import *
from oldmodels.tasks import MultiTaskModel

def GetModel(args):
    if args.model == 'resnet50_imagenet':
        backbone = resnet_imagenet(args, 50)
    elif args.model == 'densenet121_imagenet':
        backbone = densenet121_imagenet(args)
    elif args.model == 'vit_small_p16_pathology':
        backbone = vit_small_p16_pathology(args)
    elif args.model == 'swin_imagenet':
        backbone = swin_imagenet(args)
    elif args.model == 'vit_base_p16_medcoss':
        backbone = vit_base_p16_medcoss(args)
    elif args.model == 'vit_base_p14_hibou' or args.model == 'vit_base_p16_hibou':
        backbone = vit_base_hibou(args)
    elif args.model == 'vit_base_p16_uni':
        backbone = vit_base_p16_uni(args)
    elif args.model == 'vit_base_p16_conch':
        backbone = vit_base_p16_conch(args)
    else:
        raise ValueError("model not supported")
    
    model = MultiTaskModel(backbone, 768, args)
    return model 