BSZ=1024
GPU=$1
total_steps=$2
seed=$3
MULTI_TASK_STR=$4
name=$5
pareto_option=$6
DATASET=pubmed
DIM1=512
DIM2=256
ROOT_CHECKPOINT="SSL_training_${seed}_link"


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
--batch_size_multiplier_minsg 5 \
--batch_size_multiplier_ming 0 \
--khop_ming 3 \
--khop_minsg 3 \
--lr 1e-3 \
--optim adamw \
--scheduler fixed \
--weight_decay 1e-5 \
--temperature_gm 0.2 \
--temperature_minsg 0.1 \
--sub_size 256 \
--decor_size 10000 \
--decor_lamb 1e-3 \
--hid_dim ${DIM1} ${DIM2} \
--predictor_dim 512 \
--n_layer 2 \
--dropout 0. \
--seed ${seed} \
--hetero_graph_path ../hetero_graphs \
--tasks ${MULTI_TASK_STR} \
${pareto_option} \
--use_prelu \
--mask_edge \
--tvt_addr ../links/${DATASET}_tvtEdges.pkl 