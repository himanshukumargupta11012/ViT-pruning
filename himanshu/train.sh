gpu_index=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2 < 10 {print $1; exit}')

export CUDA_VISIBLE_DEVICES=$gpu_index 
python hi_main.py