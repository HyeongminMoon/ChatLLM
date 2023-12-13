#!/bin/bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_LEVEL=PIX
export MAX_JOBS=16

torchrun --standalone --nproc_per_node=8 --master_port=20001 fastchat/train/train.py \
    --model_name_or_path /disk1/data/llm_weights/Llama-2-70b-hf \
    --data_path /disk1/data/llm_datasets/sharegpt_deepl_ko/ko_dataset_2.json \
    --output_dir runs/Llama-2-70b-hf-kosharegpt \
    --bf16 True \
    --max_steps 2 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1\
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed cfg/ZeRO-3.json
    # --max_grad_norm 1.0 \
    # --num_train_epochs 3 \
    # --lr_scheduler_type "cosine" \
    
   
#--fsdp "full_shard auto_wrap offload" \
#--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

# export NCCL_P2P_DISABLE=1
#--data_path /disk1/data/llm_datasets/sharegpt_deepl_ko/ko_dataset_2.json \
#./data/dummy_conversation.json
#/disk1/data/llm_weights/vicuna-13b-v1.5
#/disk1/data/llm_weights/Llama-2-7b-hf
# --evaluation_strategy "steps" \
# --eval_steps 1200 \
    
#--fsdp "full_shard auto_wrap offload" \
#--fsdp "no_shard" \
#--fsdp "shard_grad_op auto_wrap offload" \
#--fsdp "shard_grad_op auto_wrap" \

# PIX & full_shard: 35000MB, 9s/it 
# PIX & shard_grad_op: 50000MB, 4.9s/it
# PIX & shard_grad_op & offload: 40000MB, 20s/it