#! /bin/bash
{
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sdedit_video_prefix_overlap_exp.py --base configs/cogvideox_5b_lora.yaml configs/sdedit_prefix.yaml --seed $RANDOM"
echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
exit
}
