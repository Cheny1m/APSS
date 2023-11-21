# APSS(for Training): Automatically Distributed Deep Learning Parallelism Strategies Search by Self Play

APSS 是一种基于神经网络和启发式策略的深度学习模型分布式训练切分(3D parallelism)快速策略搜索算法，它结合启发式策略和训练集群环境初步生成候选策略，然后通过深度管道策略网络（DPSN）为每个候选策略提供详细的pipeline划分，采用自我对弈的对比强化学习（CRLSP）进行离线训练，无需实际数据收集和后续应用中的微调。此仓库我们使用[Mindspore](https://www.mindspore.cn/)进行实现。

----------

## Context
- [Installation](#installation)
- [Usage and Examples](#usage-and-examples)
  - [生成PP问题的验证数据集](#生成PP问题的验证数据集)
  - [执行训练](#执行训练)


## Installation
Requirements:  
 - Python >= 3.7
 - Mindspore >= 2.1.1

### Method 1: With pip
```
pip install apss
```

### Method 2: From source
```
git clone https://github.com/Cheny1m/APSS
cd APSS
pip install -e .
```

## Usage and Examples

### 生成PP问题的验证数据集
```
python generate_pp_data.py --data_dir ./data --name validation --problem pp --graph_sizes 8 --dataset_size 10000
```
其中 `--graph_sizes 8` 代表PP问题的大小，需要与具体训练中的`graph_sizes`对应。执行上述代码后，你应该可以看到一些数据在data目录下生成。

### 执行训练

```
python run_mc.py --problem pp --graph_size 8 --num_split 3 --node_size 8  --model attention --baseline rollout --run_name 'pp_8_4' --batch_size 64 --epoch_size 1280  --val_size 10000 --eval_batch_size 1024 --val_dataset data/pp/pp_8_4_validation_seed1234.pkl
```
`graph_size` , `num_split` , `node_size` 三个命令行参数共同描述了所训练问题的大小，可根据需求动态调整。执行训练后，`.ckpt`保存在output文件夹下，日志保存在log文件夹下，可以通过tensorboard_logger在浏览器中实时查看训练数据。
