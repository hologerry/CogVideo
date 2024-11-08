#! /bin/bash
{
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sdedit_video_zero123_pi2v_onetwothree.py --base configs/cogvideox_5b_lora_prefixi2v.yaml configs_sdedit/sdedit_zero123_prefixi2v_blue.yaml --seed $RANDOM"
echo ${run_cmd}
eval ${run_cmd}


echo "DONE on `hostname`"
exit
}
