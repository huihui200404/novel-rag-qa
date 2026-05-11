FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 验证 chainlit 和其他核心包是否正确安装（构建失败会立即报错）
RUN python -c "import chainlit, langchain, chromadb; print('All critical packages OK.')"

# 复制整个项目代码（包括 chroma_db 向量库）
COPY . .

# 设置 Chainlit 监听的 ip 和端口
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=7860

# 暴露端口给外部
EXPOSE 7860

# 启动命令：运行 Chainlit 应用，监听 7860 端口
CMD ["chainlit", "run", "app.py", "--port", "7860"]