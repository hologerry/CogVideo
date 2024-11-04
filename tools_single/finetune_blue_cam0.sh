#! /bin/bash

echo "RUN on `hostname`"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python train_video.py --base configs/cogvideox_5b_lora.yaml configs/sft_blue_cam0.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
