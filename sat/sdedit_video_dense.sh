#! /bin/bash
{
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sdedit_video_dense.py --base configs/cogvideox_2b_lora.yaml configs/sdedit.yaml --sdedit-view-start-idx $1 --sdedit-view-end-idx $2 --seed $RANDOM"
echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
exit
}
