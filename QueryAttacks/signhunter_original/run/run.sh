#!/bin/bash
CUDA=$1

CUDA_VISIBLE_DEVICES=$CUDA python3 run_attack.py > aaa.log