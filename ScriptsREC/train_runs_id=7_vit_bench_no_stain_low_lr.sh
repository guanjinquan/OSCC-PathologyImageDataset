python /mnt/home/Guanjq/BackupWork/OSCC-PathologyImageDataset/Code/main_train.py \
    --runs_id "006_REC_vit_bench_32cluster" \
    --gpu_id "0" \
    --seed 109 \
    --weight_decay 6e-5 \
    --learning_rate 5e-6 \
    --backbone_lr 1e-6 \
    --acc_step 4 \
    --batch_size 2 \
    --split_filename "split_seed=2024.json" \
    --datainfo_file "all_metadata.json" \
    --img_size 512 \
    --num_epochs 200 \
    --model "vit_small_p16_pathology" \
    --use_tasks "['REC']" 
