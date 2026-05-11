import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings   # 新
from langchain_chroma import Chroma                       # 新

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# 测试1：纯 Top-K=5
retriever_simple = vectorstore.as_retriever(search_kwargs={"k": 5})
query = "福贵为什么活着"
docs = retriever_simple.invoke(query)          # 使用 invoke 代替 get_relevant_documents
print(f"Top-5 检索到 {len(docs)} 个片段：")
for i, doc in enumerate(docs):
    print(f"{i+1}. {doc.page_content[:60]}...")

# 测试2：带 Score 阈值
retriever_thresh = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.65}
)
docs = retriever_thresh.invoke(query)
print(f"\n阈值 0.65 过滤后得到 {len(docs)} 个片段")