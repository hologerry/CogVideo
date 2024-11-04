#! /bin/bash

echo "RUN on `hostname`"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python train_image_to_video.py --base configs/cogvideox_5b_lora_prefixi2v.yaml configs/sft_blue_cam2_pi2v.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
