# AI模型推理系统 - CUDA 12.8 + PyTorch 2.7.0
# 专为GPU推理优化

# 使用CUDA 12.8基础镜像
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04



FROM ${BASE_IMAGE}

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VERSION=12.8.0 \
    YOLO_CONFIG_DIR=/tmp

# 设置工作目录
WORKDIR /app

# 安装Python 3.10和系统依赖（使用Ubuntu 22.04默认版本）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



# 升级pip
RUN python -m pip install --upgrade pip

# 复制并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p logs weight /tmp

# 暴露端口
EXPOSE 5122

# 启动命令
CMD ["python", "main.py"]