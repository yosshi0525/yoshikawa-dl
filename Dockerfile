FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

#環境構築準備
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# 依存関係などのインストール
RUN apt-get update
RUN apt-get install -y wget apt-utils curl
RUN apt-get install -y build-essential gdb lcov pkg-config \
    libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
    lzma lzma-dev tk-dev uuid-dev zlib1g-dev

# Pythonのインストール
RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz \
    && tar -xf Python-3.10.13.tgz
WORKDIR /Python-3.10.13
RUN ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall
WORKDIR /
RUN rm -rf Python-3.10.13.tgz Python-3.10.13

# PATHの追加
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

# Node.jsのインストール
RUN curl -sL https://deb.nodesource.com/setup_16.x |bash - \
    && apt-get install -y --no-install-recommends \
    nodejs

# jupyter-kiteのインストール
RUN wget https://linux.kite.com/dls/linux/current && \
    chmod 777 current && \
    sed -i 's/"--no-launch"//g' current > /dev/null && \
    ./current --install ./kite-installer

# 作業ディレクトリの指定
WORKDIR /dl

# Pythonライブラリのインストール
RUN pip install --upgrade pip
RUN pip install --upgrade --no-cache-dir \
    matplotlib \
    numpy \
    Pillow \
    black \
    isort \
    # jupyter
    'jupyterlab==3.0.14' 'jupyterlab-kite>=2.0.2'\
    'jupyterlab_code_formatter==2.0.0' \
    jupyterlab-language-pack-ja-jp \
    # PyTorch
    --extra-index-url https://download.pytorch.org/whl/cu102 torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1
