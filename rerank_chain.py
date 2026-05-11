# rerank_chain.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chat
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List
from pydantic import ConfigDict

# 初始化向量库、LLM 等
chat.init_models()

print("正在加载 Reranker 模型...")
reranker = CrossEncoder(
    'BAAI/bge-reranker-base',
    max_length=512,
    device='cpu'
)

class RerankRetriever(BaseRetriever):
    k_retrieve: int = 15
    k_rerank: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = chat._vectorstore.similarity_search(query, k=self.k_retrieve)
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_pairs[:self.k_rerank]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

rerank_retriever = RerankRetriever(k_retrieve=15, k_rerank=3)

qa_rerank = RetrievalQA.from_chain_type(
    llm=chat._llm,
    chain_type="stuff",
    retriever=rerank_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": chat.QA_PROMPT}
)

if __name__ == "__main__":
    query = "有庆是怎么死的？"
    print("问题:", query)
    result = qa_rerank.invoke({"query": query})
    print(result["result"])