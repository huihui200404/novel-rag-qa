FROM python:3.11-slim

WORKDIR /app

# 安装 git 和 huggingface-cli 所需依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装 huggingface_hub
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir huggingface_hub

# 设置模型下载目录
ENV MODELS_DIR=/app/models
ENV MODEL_NAME=bge-small-zh-v1.5

# 从 HuggingFace 官方源下载模型（注意：不加镜像，因为镜像已不稳定）
# 使用镜像可能更慢，但我们设置超时和重试
RUN mkdir -p ${MODELS_DIR}/${MODEL_NAME} && \
    huggingface-cli download BAAI/${MODEL_NAME} \
    --local-dir ${MODELS_DIR}/${MODEL_NAME} \
    --local-dir-use-symlinks False \
    --resume-download \
    || (echo "Retrying with mirror..." && \
        HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/${MODEL_NAME} \
        --local-dir ${MODELS_DIR}/${MODEL_NAME} \
        --local-dir-use-symlinks False \
        --resume-download)

# 安装项目依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 验证模型文件存在
RUN test -f ${MODELS_DIR}/${MODEL_NAME}/config.json || (echo "Model download failed!" && exit 1)

# 复制项目文件
COPY . .

# 设置环境变量，告诉 sentence-transformers 从这里加载模型
ENV SENTENCE_TRANSFORMERS_HOME=${MODELS_DIR}

# 在运行时，因为模型已在本地，可以启用离线模式加快加载
ENV HF_HUB_OFFLINE=1

# 暴露端口（Railway 会通过 $PORT 注入）
EXPOSE 8000

# 使用 shell 形式以便 $PORT 正确展开
CMD chainlit run app.py --port ${PORT:-8000} --host 0.0.0.0