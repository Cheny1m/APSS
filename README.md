# APSS(for Training): Automatically Distributed Deep Learning Parallelism Strategies Search by Self Play

APSS 是一种基于神经网络和启发式策略的深度学习模型分布式训练切分(3D parallelism)快速策略搜索算法，它结合启发式策略和训练集群环境初步生成候选策略，然后通过深度管道策略网络（DPSN）为每个候选策略提供详细的pipeline划分，采用自我对弈的对比强化学习（CRLSP）进行离线训练，无需实际数据收集和后续应用中的微调。此仓库我们使用[Mindspore](https://www.mindspore.cn/)进行实现。

----------

## Context
- [Installation](#installation)
- [Usage and Examples](#usage-and-examples)
  - [生成PP问题的验证数据集](#生成PP问题的验证数据集)
  - [执行训练](#执行训练)
- [How It Works](#how-it-works)


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

### 一步执行训练

```
python -m apss.training.apss_run --graph_size 8 --num_split 3 --rebuild_data
```
* `graph_size` , `num_split` 分别代表了问题的层数大小和需要执行pipeline划分的数量，两个命令行参数共同描述了所训练问题的大小，可根据需求动态调整。
* `rebuild_data` 表示是否在执行训练前，从Data Synthesizer中生成训练数据，默认建议开启。如果需要从`.ckpt`中接续训练或无需改变之前生成的训练数据直接禁用`--rebuild_data`参数即可。训练数据可在/data目录下找到。
* 已经完成过执行训练后，`.ckpt`保存在/output文件夹下，日志保存在/log文件夹下，可以通过tensorboard_logger在浏览器中实时查看训练过程及其数据。

## How It Works
![The pipeline of APSS.](docs/apss_pipeline.png)
