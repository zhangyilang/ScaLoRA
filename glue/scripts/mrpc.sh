#!/bin/bash

SEED=0
TASK_NAME=mrpc

export CUDA_VISIBLE_DEVICES=3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=${SEED}
export TQDM_DISABLE=1

nohup torchrun \
    --nnodes=1 --nproc-per-node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    run_glue.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --task_name ${TASK_NAME} \
    --do_train --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-3 \
    --num_train_epochs 10 --warmup_ratio 0.1 \
    --cls_dropout 0 --weight_decay 0.01 \
    --output_dir ./output/${TASK_NAME}/ --overwrite_output_dir \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --lora_r 4 --lora_alpha 8 \
    --target_modules query_proj,key_proj,value_proj,output.dense,intermediate.dense \
    --seed ${SEED} \
    --scale_lora True --scale_warmup 10 \
    &> logs/${TASK_NAME}/ScaLoRA-seed${SEED}.log &
