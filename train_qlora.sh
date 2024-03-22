#!/bin/bash
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
# export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_LEVEL=PIX
export MAX_JOBS=64

deepspeed --master_port=16000 --include localhost:0,1,2,3,4,5,6,7 fastchat/train/train_lora_custom.py \
    --model_name_or_path /data/llm_weights/custom_trained/M-DIE-M-10.7B_gpt4_ep3/ \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
    --data_path "/data/llm_datasets/sftv5/deduped/korea_spelling_correction.json" "/data/llm_datasets/sftv5/v4_repeat/sharegpt_V3_format_translation(enko)-10000.json" "/data/llm_datasets/sftv5/rebal/kocommercial_mid_word_infer.json" "/data/llm_datasets/sftv5/rebal/mr_tydi_akward_correction.json" "/data/llm_datasets/sftv5/v4_repeat/naver-news-summarization.json" "/data/llm_datasets/sftv5/v4_repeat/koalpaca_v1.1-vicuna.json" "/data/llm_datasets/sftv5/deduped/wiki_qa_mid_key_infer.json" "/data/llm_datasets/sftv5/rebal/kocommercial_rear_infer.json" "/data/llm_datasets/sftv5/v4_repeat/alpaca-gpt4-korean_dedup2.json" "/data/llm_datasets/sftv5/v4_repeat/wizardlm_orca_vicuna_dedup2.json" "/data/llm_datasets/sftv5/rebal/kocommercial_rearrange.json" "/data/llm_datasets/sftv5/deduped/wiki_qa_front_infer.json" "/data/llm_datasets/sftv5/v4_repeat/aihub_law_summary.json" "/data/llm_datasets/sftv5/v4_repeat/korquad-chat-vicuna.json" "/data/llm_datasets/sftv5/v4_repeat/sharegpt_V3_format_translation(koen)-10000.json" "/data/llm_datasets/sftv5/deduped/aihub_tech_sci_summary_infer.json" "/data/llm_datasets/sftv5/v4_repeat/aihub_book_summary.json" "/data/llm_datasets/sftv5/deduped/aihub_interface_spelling_correction.json" "/data/llm_datasets/sftv5/rebal/kocommercial_rearrange_sen.json" \
    --output_dir runs/DIE-10_7B_sftv5_task \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --bf16 True \
    --evaluation_strategy "no" \
    --eval_steps 1000000  \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 12 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --q_lora True \
    --deepspeed cfg/ZeRO-2-no_offload.json \
    --gradient_checkpointing True \
    --flash_attn True \
    --max_grad_norm 1.0 \
    --lazy_preprocess True \
    --data_format 'chat-orca' \
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
