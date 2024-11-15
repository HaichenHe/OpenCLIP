#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi



python -m torch.distributed.launch --master_port 1240 --nproc_per_node=1 \
    tools/test_raw_clip_zeroshot.py --config ${config} ${@:2}