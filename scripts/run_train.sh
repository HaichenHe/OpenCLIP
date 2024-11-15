#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
NCCL_DEBUG=INFO torchrun  --nproc_per_node=1  --master_port=1238 tools/train.py  --config ${config} --log_time $now