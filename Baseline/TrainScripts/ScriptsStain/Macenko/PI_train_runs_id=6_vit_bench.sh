<<<<<<< Updated upstream
python  ./Baseline/main_train.py \
    --runs_id "macenko_006_PI_vit_bench" \
    --gpu_id "0" \
=======
python  ./Baseline/main_train.py \
    --runs_id "macenko_006_PI_vit_bench_32cluster" \
    --gpu_id "1" \
>>>>>>> Stashed changes
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
    --augment_method 'macenko' \
    --use_tasks "['PI']" \
    
