#!/bin/bash

SEED=0
LORA_R=8
LORA_ALPHA=16
LR=8e-5
TASK=$1     # boolq piqa social_iqa hellaswag winogrande arc_easy arc_challenge openbookqa
OUTPUT_DIR=./output/llama3-8b/${TASK}/scalora
LOG_DIR=./logs/llama3-8b/${TASK}

export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=${SEED}
export TQDM_DISABLE=1

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

nohup torchrun --nnodes=1 --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
	finetune_and_eval.py \
	--base_model 'meta-llama/Meta-Llama-3-8B' \
	--data_path ${TASK} \
	--output_dir ${OUTPUT_DIR} \
	--batch_size 16  --micro_batch_size 8  --num_epochs 3 \
	--dtype fp16  --seed ${SEED} \
	--cutoff_len 256  --val_set_size 0 \
	--learning_rate ${LR} \
	--adapter_name lora  --lora_r ${LORA_R}  --lora_alpha ${LORA_ALPHA} \
	--lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
	--scale_lora --scale_warmup 10 --scale_interval 10 \
	&> ${LOG_DIR}/ScaLoRA_r${LORA_R}-${LR}.log &
