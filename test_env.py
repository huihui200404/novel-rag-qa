import os
from dotenv import load_dotenv

load_dotenv()  # 自动读取 .env 文件

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"密钥加载成功，前缀为：{api_key[:10]}...")
else:
    print("密钥加载失败，请检查 .env 文件")

# 尝试导入 openai，不调用 API
import openai
print(f"openai 库版本：{openai.__version__}")