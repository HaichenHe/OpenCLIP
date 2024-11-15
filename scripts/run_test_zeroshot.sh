#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

python -m torch.distributed.launch --master_port 1240 --nproc_per_node=1 \
    test_zeroshot.py --config ${config} --weights ${weight} ${@:3}