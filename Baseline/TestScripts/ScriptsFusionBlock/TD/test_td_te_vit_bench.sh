python ./Baseline/main_test.py \
python ./Baseline/main_train.py \
    --runs_id "fusion_TE_004_TD_vit_bench" \
    --gpu_id "3" \
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
    --model "vit_small_p16_pathology" \
    --use_tasks "['TD']" \
    --fusion_block 'transformer_encoder'
