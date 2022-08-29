#!/bin/bash
CUDA=$1

CUDA_VISIBLE_DEVICES=$CUDA python run_attack.py --model=resnext101_denoise --dataset=imagenet \
    --batch_size=100 --fd_eta=0.1 --max_queries=2500 --image_lr=0.01 \
    --mode="linf" --online_lr=100 --exploration=1.0 --epsilon=0.015625 --gradient_iters=1 \
    --total_images=1000 --tile_size=50 --log_progress