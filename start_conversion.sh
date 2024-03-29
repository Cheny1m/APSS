#!/bin/bash
# 启动命令
# bash start_conversion.sh

# 预设参数
PTH_PATH="/root/APSS/torchoutputs/pp_30/v2_pp30_8_rollout_20230405T223800/"   # /root/APSS/torchoutputs/pp_30/
PTH_NAME="/root/APSS/torchoutputs/pp_30/v2_pp30_8_rollout_20230405T223800/epoch-99.pt"
NUM_SPLIT=7  # [1,3,7,15]


# 无需设置
NODE_SIZE=30 # [30]
MODEL_VERSION="v2"
NUM_SPLIT_PLUS_ONE=$((NUM_SPLIT + 1))
CKPT_PATH="/root/APSS/checkpoint/pp_${NODE_SIZE}_${NUM_SPLIT_PLUS_ONE}_${MODEL_VERSION}"
CKPT_NAME="${CKPT_PATH}/pp_${NODE_SIZE}_${NUM_SPLIT_PLUS_ONE}_final"   # /root/APSS/resource/outputs/pp_NODE_SIZE/pp_NODE_SIZE_NUM_SPLIT_final
if [ ! -d "$CKPT_PATH" ]; then
    mkdir "$CKPT_PATH"
    echo "文件夹已创建：$CKPT_PATH"
else
    echo "文件夹已存在：$CKPT_PATH"
fi

# 调用Python脚本并传入预设参数
python pt2ckpt.py "$PTH_NAME" "$CKPT_NAME" "$NUM_SPLIT" "$NODE_SIZE"

# 复制config
cp "$PTH_PATH/args.json" "$CKPT_PATH"

# 确认文件已经被复制
if [ $? -eq 0 ]; then
    echo "文件已经成功复制到$CKPT_PATH。"
else
    echo "复制文件时出错。"
    exit 1
fi



