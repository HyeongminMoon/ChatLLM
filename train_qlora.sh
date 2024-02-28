#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_LEVEL=PIX
export MAX_JOBS=16

deepspeed --master_port=16000 --include localhost:0,1,2,3 fastchat/train/train_lora_custom.py \
    --model_name_or_path /workspaces/disk0/data/llm_weights/MoMo-72B-lora-1.8.7-DPO \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
    --data_path "/data/llm_datasets/custom/vicuna_format/gpt_evol_1.3k-vicuna.json" \
    --output_dir runs/PIE-72B_v0.2_gpt_evol \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --evaluation_strategy "no" \
    --eval_steps 1000000  \
    --save_strategy "epoch" \
    --save_steps 5000 \
    --save_total_limit 12 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --q_lora True \
    --deepspeed cfg/ZeRO-2-no_offload.json \
    --gradient_checkpointing True \
    --flash_attn True \
    --max_grad_norm 1.0 \
    --lazy_preprocess True \
    --data_format 'qwen' \
    --load_in_8bit False \
    --padding_side "right"

# 7b, batch size1, 23000MB, 23h
# 7b, batch size1, flash_attn, ZeRO-2, 23000MB, 11h
# 7b, batch size1, flash_attn, ZeRO-3, ???, ??? 
# 7b, batch size1, flash_attn, ZeRO-2, 8bit, 23000MB, , 18h

# 13b, batch size4, flash_attn, ZeRO-1, 48044MB, 19h
# 13b, batch size4, flash_attn, ZeRO-2, 48433MB, 19h
# 13b, batch size4, flash_attn, ZeRO-3, 31310MB, 80h

# 70b, batch size1, flash_attn, ZeRO-2, 8bit, ??? *진행중 -> RuntimeError: shape '[1, 4096, 64, 128]' is invalid for input of size 4194304
# 70b, batch size1, ZeRO-2, 8bit, ???  *OOM
# 70b, batch size1, ZeRO-3, 8bit, ???  *OOM
# 70b, batch size1, ZeRO-3, ??? *OOM

# 70b, batch size1, FSDP, Failed     


## --deepspeed cfg/ZeRO-2-no_offload.json \
## --deepspeed cfg/ZeRO-3-no_offload.json \


#lora는 scheduler constant로?
