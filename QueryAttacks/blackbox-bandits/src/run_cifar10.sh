#!/bin/bash
CUDA=$1

CUDA_VISIBLE_DEVICES=$CUDA python3 run_attack.py --model=Standard --defense=AAAR --dataset=cifar10 \
    --batch_size=1000 --fd_eta=0.1 --max_queries=2500 --image_lr=0.01 \
    --mode="linf" --online_lr=100 --exploration=1.0 --epsilon=0.03125 --gradient_iters=1 \
    --total_images=10000 --tile_size=50 --log_progress