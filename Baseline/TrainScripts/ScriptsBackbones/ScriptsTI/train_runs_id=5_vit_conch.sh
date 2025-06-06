python ./Baseline/main_train.py \
    --runs_id "005_TI_vit_base_p16_conch" \
    --gpu_id "1" \
    --seed 109 \
    --weight_decay 0.01 \
    --learning_rate 1e-6 \
    --backbone_lr 5e-7 \
    --acc_step 16 \
    --batch_size 1 \
    --split_filename "split_seed=2024.json" \
    --datainfo_file "all_metadata.json" \
    --img_size 512 \
    --optimizer "AdamW" \
    --num_epochs 400 \
    --model "vit_base_p16_conch" \
    --use_tasks "['TI']"
