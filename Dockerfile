FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 验证核心包是否安装成功
RUN python -c "import chainlit, langchain, chromadb; print('All critical packages OK.')"

COPY . .

# 不再设置任何 CHAINLIT_HOST 或 CHAINLIT_PORT 环境变量
# 不写 EXPOSE，Railway 会自动处理

# 启动命令：通过 shell 形式让 $PORT 被替换
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0"]