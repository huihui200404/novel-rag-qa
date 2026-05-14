FROM python:3.11-slim

WORKDIR /app

# 将本地模型直接复制进镜像（请确保 local_model/bge-small-zh-v1.5 存在）
COPY local_model/bge-small-zh-v1.5 /app/models/bge-small-zh-v1.5
RUN test -f /app/models/bge-small-zh-v1.5/config.json || (echo "模型文件缺失！" && exit 1)

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/models/bge-small-zh-v1.5
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

# ★ 关键修改：使用 JSON 数组格式，Chainlit 直接成为 PID 1
CMD ["chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]