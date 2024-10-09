# tmux - 1
python /home/Guanjq/Work/OSCC-PathologyImageDataset/Baseline/main_test.py \
    --runs_id "003_SUM_vit_bench_32cluster" \
    --gpu_id "0" \
    --seed 109 \
    --weight_decay 6e-5 \
    --learning_rate 1e-6 \
    --backbone_lr 5e-7 \
    --acc_step 8 \
    --batch_size 2 \
    --split_filename "split_seed=2024.json" \
    --datainfo_file "all_metadata.json" \
    --img_size 512 \
    --num_epochs 400 \
    --model "vit_small_p16_pathology" \
    --use_tasks "['REC', 'LNM', 'TD', 'TI', 'CE', 'PI']"
