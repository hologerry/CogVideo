#! /bin/bash

echo "RUN on $(hostname)"

run_cmd="torchrun --standalone --nproc_per_node=4 train_video.py --base configs/cogvideox_5b_lora.yaml configs/sft_red.yaml --seed $RANDOM"

# run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train_video.py --base configs/test_cogvideox_5b_i2v_lora.yaml configs/test_sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
