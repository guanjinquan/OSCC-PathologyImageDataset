# python /mnt/home/Guanjq/BackupWork/OSCC-PathologyImageDataset/Baseline/main_train_grad_norm.py \
#     --runs_id "001_GradNorm_vit_bench_32cluster" \
#     --gpu_id "1" \
#     --seed 109 \
#     --weight_decay 6e-5 \
#     --learning_rate 1e-6 \
#     --backbone_lr 5e-7 \
#     --acc_step 8 \
#     --batch_size 2 \
#     --split_filename "split_seed=2024.json" \
#     --datainfo_file "all_metadata.json" \
#     --img_size 512 \
#     --num_epochs 200 \
#     --model "vit_small_p16_pathology" \
#     --use_tasks "['REC', 'LNM', 'TD', 'TI', 'CE', 'PI']"

python3 -m torch.distributed.launch \
    --use_env \
    --master_port 23334 \
    --nproc_per_node=2 \
    /mnt/home/Guanjq/BackupWork/OSCC-PathologyImageDataset/Baseline/main_train_grad_norm.py \
    --runs_id "001_GradNorm_vit_bench_32cluster" \
    --gpu_id "0,1" \
    --seed 109 \
    --weight_decay 6e-5 \
    --learning_rate 1e-6 \
    --backbone_lr 5e-7 \
    --acc_step 4 \
    --batch_size 4 \
    --split_filename "split_seed=2024.json" \
    --datainfo_file "all_metadata.json" \
    --img_size 512 \
    --num_epochs 200 \
    --model "vit_small_p16_pathology" \
    --use_tasks "['REC', 'LNM', 'TD', 'TI', 'CE', 'PI']" \
    --use_ddp True \
    --use_amp True \
