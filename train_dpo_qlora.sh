#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=PIX
export MAX_JOBS=16

deepspeed --master_port=11666 --include localhost:0,1,2,3,4,5,6,7 fastchat/train/train_dpo_lora.py \
    --model_name_or_path /data/llm_weights/custom_trained/PIE-72B-45000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj \
    --data_path "/data/llm_datasets/custom/ados/dpo/ados_dpo_v2.json" \
    --output_dir runs/PIE-72B-45000_dpo \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --evaluation_strategy "no" \
    --eval_steps 1000000  \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 12 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 3072 \
    --q_lora True \
    --deepspeed cfg/ZeRO-2-no_offload.json \
    --gradient_checkpointing True \
    --flash_attn True \
    --max_grad_norm 1.0 \
    --data_format 'qwen' \
    --beta 0.1 \
    --max_length 3072 \
    --max_prompt_length 3072 \
    --max_target_length 3072 \
    --padding_side "right"
    
# 7b, batch size1, 17201MB, 24h
# 7b, batch size1, flash_attn, 8889MB, 13h
# 13b, batch size4, 71155MB, 38h
# 70b, batch size1, 69311MB, 160h
# 70b, batch size2, OOM
# custom/merged_korean_datasets-vicuna-v2.json


