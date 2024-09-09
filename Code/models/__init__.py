from models.backbones import *
from models.tasks import MultiTaskModel, metrics

def GetModel(args):
    
    if args.model == 'check_pathology':
        print("check_pathology model")
        return vit_small_p16_pathology_model(args)

    if args.model == 'resnet50_pathology':
        backbone = resnet50_pathology(args)
    elif  args.model == 'resnet34_imagenet':
        backbone = resnet_imagenet(args, 34)
    elif args.model == 'resnet50_imagenet':
        backbone = resnet_imagenet(args, 50)
    elif args.model == 'densenet121_imagenet':
        backbone = densenet121_imagenet(args)
    elif args.model == 'densenet161_pathology':
        backbone = densenet161_pathology(args)
    elif args.model == 'vit_small_p16_pathology':
        backbone = vit_small_p16_pathology(args)
    elif args.model == 'vit_base_p32_imagenet':
        backbone = vit_base_p32_imagenet(args)
    elif args.model == 'swin_imagenet':
        backbone = swin_imagenet(args)
    elif args.model == 'vit_base_p32_no_pre':
        backbone = vit_base_p32_no_pre(args)
    elif args.model == 'densenet121_no_pre':
        backbone = densenet121_no_pre(args)
    elif args.model == 'swin_no_pre':
        backbone = swin_no_pre(args)
    elif args.model == "vit_small_p16_imagenet":
        backbone = vit_small_p16_imagenet(args)
    elif args.model == "resnext50_imagenet":
        backbone = resnext50_imagenet(args)
    elif args.model == 'vit_base_p16_medcoss':
        backbone = vit_base_p16_medcoss(args)
    elif args.model == 'vit_base_p14_hibou' or args.model == 'vit_base_p16_hibou':
        backbone = vit_base_hibou(args)
    elif args.model == 'vit_base_p16_imagenet':
        backbone = vit_base_p16_imagenet(args)
    else:
        raise ValueError("model not supported")
    
    model = MultiTaskModel(backbone, 768, args)
    return model 