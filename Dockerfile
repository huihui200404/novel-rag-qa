FROM python:3.11-slim

WORKDIR /app

# 安装必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 huggingface-cli
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir huggingface_hub

# 设置模型保存路径
ENV MODELS_DIR=/app/models
ENV MODEL_NAME=bge-small-zh-v1.5

# 从官方源下载模型（重试 5 次，每次间隔 30 秒）
RUN mkdir -p ${MODELS_DIR}/${MODEL_NAME} && \
    huggingface-cli download BAAI/${MODEL_NAME} \
        --local-dir ${MODELS_DIR}/${MODEL_NAME} \
        --local-dir-use-symlinks False \
        --resume-download \
        --max-retries 5 \
        --retry-delay 30 \
    && echo "模型下载成功"

# 确认关键文件存在
RUN test -f ${MODELS_DIR}/${MODEL_NAME}/config.json || (echo "模型文件缺失！" && exit 1)

# 安装项目依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=${MODELS_DIR}/${MODEL_NAME}
ENV SENTENCE_TRANSFORMERS_HOME=${MODELS_DIR}
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD chainlit run app.py --port ${PORT:-8000} --host 0.0.0.0