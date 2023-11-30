FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

#環境構築準備
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# 依存関係などのインストール
RUN apt-get update
RUN apt-get install -y wget apt-utils
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

# 作業ディレクトリの指定
WORKDIR /dl

# Pythonライブラリのインストール
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
