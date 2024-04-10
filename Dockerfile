# 使用CUDA 11.1的官方基础镜像
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

# 设置非交互式安装，防止在构建过程中出现询问
ENV DEBIAN_FRONTEND=noninteractive

# 安装必要的系统依赖项
RUN apt update && apt install -y \
 wget \
 build-essential \
 zlib1g-dev \
 libncurses5-dev \
 libgdbm-dev \
 libnss3-dev \
 libssl-dev \
 libreadline-dev \
 libffi-dev \
 libsqlite3-dev \
 libbz2-dev \
 liblzma-dev \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

# 准备构建 Python 的临时目录
WORKDIR /temp

# 下载并解压 Python 源代码，然后编译安装
RUN wget https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tgz \
 && tar -xzf Python-3.9.10.tgz \
 && cd Python-3.9.10 \
 && ./configure --enable-optimizations \
 && make -j 8 \
 && make altinstall \
 && cd .. \
 && rm -rf Python-3.9.10.tgz Python-3.9.10

# 更新 Python 和 Pip 到新安装的版本
RUN ln -sf /usr/local/bin/python3.9 /usr/bin/python3 \
 && ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3 \
 && pip3 --no-cache-dir install --upgrade pip \
 && echo "alias python='/usr/local/bin/python3.9'" >> ~/.bashrc \
 && echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc

# 设置工作目录
WORKDIR /workspace

# 安装MindSpore GPU版本
RUN python3.9 -m pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置PATH环境变量
ENV PATH="/usr/local/bin:$PATH"

# 安装其他依赖
RUN pip3 install tensorboard_logger tqdm numpy
