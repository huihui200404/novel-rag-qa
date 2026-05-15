import os

# 本地 Embedding 模型路径（本地开发用项目根下的 local_model）
_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "local_model", "bge-small-zh-v1.5")
# Docker 部署时的路径（Railway 会把模型复制到 /app/models/）
_DOCKER_DIR = "/app/models/bge-small-zh-v1.5"

# 优先使用 Docker 路径，如果不存在则退回本地路径
LOCAL_MODEL_PATH = _DOCKER_DIR if os.path.isdir(_DOCKER_DIR) else _LOCAL_DIR