python /mnt/home/Guanjq/BackupWork/OSCC-PathologyImageDataset/Code/main_train.py \
    --runs_id "002_TI_resnet50_pathology" \
    --gpu_id "1" \
    --seed 109 \
    --weight_decay 6e-5 \
    --learning_rate 1e-6 \
    --backbone_lr 5e-7 \
    --acc_step 8 \
    --batch_size 2 \
    --split_filename "split_seed=2024.json" \
    --datainfo_file "all_metadata.json" \
    --img_size 512 \
    --num_epochs 200 \
    --model "resnet50_pathology" \
    --use_tasks "['TI']" \
    --augment_method 'vahadane' \
    --stain_prob 0.5
