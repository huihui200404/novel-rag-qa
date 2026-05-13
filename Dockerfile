FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 验证核心包
RUN python -c "import chainlit, langchain, chromadb; print('All critical packages OK.')"

# 复制项目所有文件
COPY . .

# 强制设置端口为 8080（这是 Railway 给你的固定端口）
ENV CHAINLIT_PORT=8080
EXPOSE 8080

# 启动应用（使用 exec 形式，确保进程 PID 为 1，打印日志完整）
CMD ["chainlit", "run", "app.py", "--port", "8080", "--host", "0.0.0.0"]