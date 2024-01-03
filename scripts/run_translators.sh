#!/bin/bash
export NCCL_P2P_LEVEL=PIX


export CUDA_VISIBLE_DEVICES=0
nohup python fastchat/serve/vllm_worker.py --model-path /workspaces/disk0/data/llm_weights/Gugugo-koen-7B-V1.1/ --controller-address http://localhost:21001 --port 31002 --worker-address http://localhost:31002 --num-gpus 1 --max-num-batched-tokens 6120 --host 0.0.0.0 --limit-worker-concurrency 1024 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup python fastchat/serve/vllm_worker.py --model-path /workspaces/disk0/data/llm_weights/Gugugo-koen-7B-V1.1/ --controller-address http://localhost:21001 --port 31003 --worker-address http://localhost:31003 --num-gpus 1 --max-num-batched-tokens 6120 --host 0.0.0.0 --limit-worker-concurrency 1024 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=2
nohup python fastchat/serve/vllm_worker.py --model-path /workspaces/disk0/data/llm_weights/Gugugo-koen-7B-V1.1/ --controller-address http://localhost:21001 --port 31004 --worker-address http://localhost:31004 --num-gpus 1 --max-num-batched-tokens 6120 --host 0.0.0.0 --limit-worker-concurrency 1024 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=3
nohup python fastchat/serve/vllm_worker.py --model-path /workspaces/disk0/data/llm_weights/Gugugo-koen-7B-V1.1/ --controller-address http://localhost:21001 --port 31005 --worker-address http://localhost:31005 --num-gpus 1 --max-num-batched-tokens 6120 --host 0.0.0.0 --limit-worker-concurrency 1024 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=4
nohup python fastchat/serve/vllm_worker.py --model-path /workspaces/disk0/data/llm_weights/Gugugo-koen-7B-V1.1/ --controller-address http://localhost:21001 --port 31006 --worker-address http://localhost:31006 --num-gpus 1 --max-num-batched-tokens 6120 --host 0.0.0.0 --limit-worker-concurrency 1024 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=5
nohup python fastchat/serve/vllm_worker.py --model-path /workspaces/disk0/data/llm_weights/Gugugo-koen-7B-V1.1/ --controller-address http://localhost:21001 --port 31007 --worker-address http://localhost:31007 --num-gpus 1 --max-num-batched-tokens 6120 --host 0.0.0.0 --limit-worker-concurrency 1024 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=6
nohup python fastchat/serve/vllm_worker.py --model-path /workspaces/disk0/data/llm_weights/Gugugo-koen-7B-V1.1/ --controller-address http://localhost:21001 --port 31008 --worker-address http://localhost:31008 --num-gpus 1 --max-num-batched-tokens 6120 --host 0.0.0.0 --limit-worker-concurrency 1024 > /dev/null 2>&1 &
