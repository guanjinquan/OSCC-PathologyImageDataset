python ./Baseline/main_test.py \
    --runs_id "seed42_005_PI_vit_base_p16_conch" \
    --gpu_id "0" \
    --seed 42 \
    --weight_decay 6e-5 \
    --learning_rate 1e-6 \
    --backbone_lr 5e-7 \
    --acc_step 16 \
    --batch_size 1 \
    --split_filename "split_seed=2024.json" \
    --datainfo_file "all_metadata.json" \
    --img_size 512 \
    --num_epochs 400 \
    --model "vit_base_p16_conch" \
    --use_tasks "['PI']"
