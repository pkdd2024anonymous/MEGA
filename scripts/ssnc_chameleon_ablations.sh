#!/bin/bash

BSZ=1024
GPU=0
total_steps=10001
seed=2345
MULTI_TASK_STR="p_link p_recon p_ming p_decor p_minsg"
pareto_option=" "
DATASET=chameleon
ROOT_CHECKPOINT="ABL_SSL_training_${seed}_ssnc"

# Ablation studies configurations
declare -a ablation_configs=(
  "1024_256"
  "2024_256"
  #  "2024_2024_256"
  #  "1024_1024_1024_1024_1024_256"
  #  "512_512_512_512_512_512_512_256"
  #  


)

for config in "${ablation_configs[@]}"; do
    IFS='_' read -r -a dims <<< "$config"
    name="pareto_${config}"

    echo "======================="
    echo "Name: ${name}"
    echo "Hid dims: ${dims[@]}"


    CUDA_VISIBLE_DEVICES=${GPU} python ../pretrain_model.py \
    --world_size 1 \
    --worker 8 \
    --name ${name} \
    --checkpoint_dir "${ROOT_CHECKPOINT}/${DATASET}" \
    --dataset ${DATASET} \
    --split random \
    --pretrain_label_dir ../pretrain_labels \
    --total_steps ${total_steps} \
    --warmup_step 100 \
    --per_gpu_batch_size ${BSZ} \
    --batch_size_multiplier_minsg 1 \
    --batch_size_multiplier_ming 1 \
    --khop_ming 3 \
    --khop_minsg 3 \
    --lr 5e-5 \
    --optim adamw \
    --scheduler fixed \
    --weight_decay 1e-5 \
    --temperature_gm 0.2 \
    --temperature_minsg 0.1 \
    --sub_size 256 \
    --decor_size 1500 \
    --decor_lamb 1e-3 \
    --hid_dim ${dims[@]} \
    --predictor_dim 512 \
    --n_layer 2 \
    --dropout 0. \
    --seed ${seed} \
    --hetero_graph_path ../hetero_graphs \
    --tasks ${MULTI_TASK_STR} \
    ${pareto_option} \
    --use_prelu

done

