from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
try:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=["测试文本"]
    )
    print("Embedding 调用成功，向量维度：", len(response.data[0].embedding))
except Exception as e:
    print("调用失败：", e)