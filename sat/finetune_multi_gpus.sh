#! /bin/bash

echo "RUN on $(hostname)"

run_cmd="torchrun --standalone --nproc_per_node=4 train_video.py --base configs/cogvideox_5b_lora.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
