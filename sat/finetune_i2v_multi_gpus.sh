#! /bin/bash

echo "RUN on $(hostname)"

run_cmd="torchrun --standalone --nproc_per_node=4 train_image_to_video.py --base configs/cogvideox_5b_lora_i2v.yaml configs/sft_i2v.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
