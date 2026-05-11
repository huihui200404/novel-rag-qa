# retrieval_compare.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"          # 必须在所有导入之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chat                                  # 用 import 模块，而非 from import 变量
chat.init_models()                           # 初始化完成

# 现在可以安全、动态地访问 chat._vectorstore
vectorstore = chat._vectorstore

from langchain_community.retrievers import BM25Retriever   # 新路径，消除弃用警告
from langchain.retrievers import EnsembleRetriever

# 准备全量文本
all_texts = [doc for doc in vectorstore.get()["documents"]]

# 构建三种检索器
bm25 = BM25Retriever.from_texts(all_texts)
bm25.k = 7

vec_ret = vectorstore.as_retriever(search_kwargs={"k": 7})
ensemble_ret = EnsembleRetriever(
    retrievers=[bm25, vec_ret],
    weights=[0.5, 0.5]
)

# 测试问题
test_queries = [
    "有庆",
    "福贵输掉了什么家产",
    "活着这本书有什么讽刺意味",
    "家珍得病",
    "苦根怎么死"
]

print("=== 检索 Top-3 对比 ===\n")
for q in test_queries:
    print(f"问题：{q}")
    bm25_docs = bm25.invoke(q)
    vec_docs = vec_ret.invoke(q)
    ens_docs = ensemble_ret.invoke(q)

    print("BM25:")
    for i, doc in enumerate(bm25_docs[:3]):
        print(f"  {i+1}. {doc.page_content[:80]}...")
    print("向量:")
    for i, doc in enumerate(vec_docs[:3]):
        print(f"  {i+1}. {doc.page_content[:80]}...")
    print("混合:")
    for i, doc in enumerate(ens_docs[:3]):
        print(f"  {i+1}. {doc.page_content[:80]}...")
    print("-" * 40)