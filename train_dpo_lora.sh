#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_P2P_LEVEL=PIX
export MAX_JOBS=16

deepspeed fastchat/train/train_dpo_lora.py \
    --model_name_or_path /workspaces/data/llm_weights/custom_trained/DIE_10.7b_v0.1_ep2 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
    --output_dir runs/DIE_10.7b_sft_v4_dpo_v2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --bf16 True \
    --evaluation_strategy "no" \
    --eval_steps 1000000  \
    --save_strategy "epoch" \
    --save_steps 2000000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --q_lora False \
    --deepspeed cfg/ZeRO-2-no_offload.json \
    --gradient_checkpointing True \
    --flash_attn True \
    --max_grad_norm 1.0 \
    --data_format 'chat-orca' \
    --beta 0.1 \
    --max_length 4096 \
    --max_prompt_length 4096 \
    --max_target_length 4096
    
# 7b, batch size1, 17201MB, 24h
# 7b, batch size1, flash_attn, 8889MB, 13h
# 13b, batch size4, 71155MB, 38h
# 70b, batch size1, 69311MB, 160h
# 70b, batch size2, OOM
# custom/merged_korean_datasets-vicuna-v2.json


