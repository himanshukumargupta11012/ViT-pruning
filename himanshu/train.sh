#!/bin/bash


read gpu_index memory < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, 'NR==1 || $2 < min {min=$2; idx=$1} END {print idx, min}')

echo "Memory used: $memory MB"

export CUDA_VISIBLE_DEVICES=$gpu_index
python hi_main.py -d $1