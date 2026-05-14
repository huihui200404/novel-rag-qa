FROM python:3.11-slim

WORKDIR /app

# 安装必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装 huggingface-cli
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir huggingface_hub

# 设置模型保存路径
ENV MODELS_DIR=/app/models
ENV MODEL_NAME=bge-small-zh-v1.5

# 下载模型到本地目录（使用官方源，失败则重试镜像）
RUN mkdir -p ${MODELS_DIR}/${MODEL_NAME} && \
    ( huggingface-cli download BAAI/${MODEL_NAME} \
        --local-dir ${MODELS_DIR}/${MODEL_NAME} \
        --local-dir-use-symlinks False \
        --resume-download \
      || ( echo "官方源下载失败，尝试使用镜像..." && \
           HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/${MODEL_NAME} \
               --local-dir ${MODELS_DIR}/${MODEL_NAME} \
               --local-dir-use-symlinks False \
               --resume-download \
         ) \
    ) && echo "模型下载成功"

# 确认模型文件存在
RUN test -f ${MODELS_DIR}/${MODEL_NAME}/config.json || (echo "模型 config.json 缺失！" && exit 1)
RUN test -f ${MODELS_DIR}/${MODEL_NAME}/model.safetensors || (echo "模型 model.safetensors 缺失！" && exit 1)

# 安装项目依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件
COPY . .

# 设置环境变量，让程序从本地加载模型
ENV MODEL_PATH=${MODELS_DIR}/${MODEL_NAME}
ENV SENTENCE_TRANSFORMERS_HOME=${MODELS_DIR}
ENV HF_HUB_OFFLINE=1

# 暴露端口
EXPOSE 8000

# 启动命令（使用 shell 形式读取 $PORT）
CMD chainlit run app.py --port ${PORT:-8000} --host 0.0.0.0