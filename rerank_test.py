import cohere
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# 1. 初始召回 20 个片段
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./huozhe_chroma", embedding_function=embedding)
query = "《活着》的书名有什么反讽意味？"
docs = vectorstore.similarity_search_with_score(query, k=20)

# 2. 准备用Rerank模型重排序
docs_texts = [doc.page_content for doc, score in docs]
rerank_response = co.rerank(
    query=query,
    documents=docs_texts,
    top_n=3,            # 最终只要3个
    model="rerank-english-v2.0" # 也可用中文版"rerank-multilingual-v2.0"
)

# 3. 获取重排序后的最佳文档
print("=== Rerank后的Top3文本 ===")
for result in rerank_response.results:
    idx = result.index
    print(f"相关度: {result.relevance_score:.4f}")
    print(docs_texts[idx][:200])
    print("---")