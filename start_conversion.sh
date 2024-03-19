#!/bin/bash

# 启动命令
# bash start_conversion.sh

# 预设参数
PTH_PATH="/root/APSS/torchoutputs/pp_30/v3_pp30_16_rollout_20230420T145616/epoch-148.pt"   # /root/APSS/torchoutputs/pp_30/
NUM_SPLIT=15  # [1,3,7,15]


# 无需设置
NODE_SIZE=30 # [30]
NUM_SPLIT_PLUS_ONE=$((NUM_SPLIT + 1))
CKPT_PATH="/root/APSS/resource/outputs/pp_${NODE_SIZE}/pp_${NODE_SIZE}_${NUM_SPLIT_PLUS_ONE}_final"   # /root/APSS/resource/outputs/pp_NODE_SIZE/pp_NODE_SIZE_NUM_SPLIT_final


# 调用Python脚本并传入预设参数
python pt2ckpt.py "$PTH_PATH" "$CKPT_PATH" "$NUM_SPLIT" "$NODE_SIZE"



