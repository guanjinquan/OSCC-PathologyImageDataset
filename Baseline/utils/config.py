import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MLP Model')

    # path settings
    parser.add_argument('--data_root', type=str, default="./Data")
    parser.add_argument('--ckpt_path', type=str, default='./Checkpoints/', help='the path to save checkpoints')
    parser.add_argument('--log_path', type=str, default='./Results', help='the path to save log')
    parser.add_argument('--load_pth_path', type=str)
    
    
    # dataset settings 
    parser.add_argument('--datainfo_file', type=str, default="pathology_info.json") 
    parser.add_argument('--split_filename', type=str, default="split_seed=2024.json")
    parser.add_argument('--data_type', type=str, default="ALL")
    parser.add_argument('--img_size', type=str, default="512", help="image size, e.g. 512,512 or 1944,2592")
    parser.add_argument('--crop_scale', type=float, default=0.8)
    parser.add_argument('--stain_prob', type=float, default=0.5)
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument('--only_grey', type=bool, default=False)
    parser.add_argument('--augment_method', type=str, default=None)
    
    
    # models settings 
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_feat', type=int, default=-1, help='number of features, -1 means use all features')
    parser.add_argument('--freezed_backbone', action="store_true", help='whether to freeze the backbone, default unfreeze')
    parser.add_argument('--input_feats', action="store_true", help='whether to freeze the backbone, default unfreeze')
    parser.add_argument('--fusion_block', type=str, default='concat', help="fusion_block: [concat, LMF, gated, transformer_encoder]")
    parser.add_argument('--use_tasks', type=str, default="['tumor_diff', 'cancer_embolus', 'invasion', 'tumor_N', 'recurrence']", help="use task, e.g. ['tumor_diff', 'cancer_embolus', 'invasion', 'tumor_N', 'recurrence']")

    
    # trainer settings
    parser.add_argument("--runs_id", type=str)
    parser.add_argument("--acc_step", type=int)
    parser.add_argument("--train_mode", type=str, default="TVT")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=109, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--backbone_lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=6e-5, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choose optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        help='choose scheduler')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    
    
    args = parser.parse_args()

    if args.img_size == "512":
        args.img_size = (512, 512)
    elif args.img_size == "512,512":
        args.img_size = (512, 512)
    elif args.img_size == "1944,2592":
        args.img_size = (1944, 2592)
    else:
        args.img_size = tuple(map(int, args.img_size.split(',')))

    return args
