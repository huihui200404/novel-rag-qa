# rag_system.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from typing import List
from langchain_core.documents import Document

# ---------- 书名映射 ----------
BOOK_MAP = {
    "活着": "huozhe",
    "许三观卖血记": "xusanguan",
}

def resolve_collection_name(name: str) -> str:
    return BOOK_MAP.get(name, name)

# ---------- 提示词模板 ----------
TEMPLATE = """# 角色
你是一位中国现当代文学研究员，专攻余华、余华相关作品。

# 任务
基于【知识库】中提供的原文片段，回答用户关于小说的问题。

# 输出格式（严格遵守）
1. **核心结论**：用一句话准确回答问题。
2. **文本依据**：直接引用知识库中出现的**具体原文**（至少两处），不要添加章节或页码，只需引用原句。
3. **深层解读**：结合人物命运或时代背景进行简短分析。

# 边界规则
- 如果知识库中的片段不足以回答问题，回复：“根据现有资料，无法确认该问题。请尝试询问具体情节或人物。”
- **严禁使用任何外部知识**，只能依据提供的上下文。
- 不允许编造章节、时间或人物关系。

# 上下文
{context}

# 用户问题
{question}
"""
QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=TEMPLATE)

# ---------- 全局缓存 ----------
_vectorstore = None
_llm = None
_embedding_model = None
_current_collection = None

def init_models(collection_name=None):
    global _vectorstore, _llm, _embedding_model, _current_collection

    target = resolve_collection_name(collection_name) if collection_name else None

    if _vectorstore is not None and _llm is not None and _current_collection == target:
        return _vectorstore, _llm

    if _embedding_model is None:
        print("正在加载 Embedding 模型...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    if not target:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        if not collections:
            raise RuntimeError("未找到任何向量库，请先运行构建脚本。")
        target = collections[0].name
        print(f"自动选择向量库: {target}")

    print(f"正在加载向量库: {target}")
    _vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=_embedding_model,
        collection_name=target
    )
    _current_collection = target
    doc_count = len(_vectorstore.get()["documents"])
    print(f"向量库 [{target}] 包含 {doc_count} 个文档片段")

    if _llm is None:
        print("正在初始化智谱 LLM...")
        _llm = ChatOpenAI(
            model="glm-4-flash",
            temperature=0,
            openai_api_key=os.getenv("ZHIPU_API_KEY"),
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
            max_tokens=2048
        )

    return _vectorstore, _llm


class SimpleEnsembleRetriever(BaseRetriever):
    """加固版混合检索器，带噪音过滤"""
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        all_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            if weight <= 0:
                continue
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                print(f"[检索器警告] {retriever} 调用失败: {e}")
                docs = []
            for doc in docs:
                doc.metadata["_weight"] = weight
            all_docs.extend(docs)

        if not all_docs:
            return []

        # 去重并计算平均权重
        unique = {}
        counts = {}
        for doc in all_docs:
            key = doc.page_content
            if key in unique:
                unique[key].metadata["_weight"] += doc.metadata["_weight"]
                counts[key] += 1
            else:
                unique[key] = doc
                counts[key] = 1

        for key in unique:
            unique[key].metadata["_weight"] /= counts[key]

        sorted_docs = sorted(unique.values(), key=lambda d: d.metadata["_weight"], reverse=True)

        # 过滤噪音片段
        noise_keywords = ["全书完", "声明", "版权", "用户上传之内容结束", "txt02.com"]
        clean = []
        for doc in sorted_docs:
            text = doc.page_content
            if len(text) < 50 and any(kw in text for kw in noise_keywords):
                continue
            if sum(text.count(kw) for kw in noise_keywords) > 2:
                continue
            clean.append(doc)
        if clean:
            sorted_docs = clean

        return sorted_docs


def build_qa_chain(collection_name=None, k=5, retrieval_type="ensemble"):
    vs, llm = init_models(collection_name)
    all_texts = vs.get()["documents"]

    if retrieval_type == "vector":
        retriever = vs.as_retriever(search_kwargs={"k": k})
    elif retrieval_type == "bm25":
        if not all_texts:
            print("[警告] 向量库为空，回退到向量检索")
            retriever = vs.as_retriever(search_kwargs={"k": k})
        else:
            bm25 = BM25Retriever.from_texts(all_texts)
            bm25.k = k
            retriever = bm25
    else:  # ensemble
        if not all_texts:
            print("[警告] 向量库为空，仅使用向量检索")
            vec_ret = vs.as_retriever(search_kwargs={"k": k})
            retriever = SimpleEnsembleRetriever(
                retrievers=[vec_ret],
                weights=[1.0]
            )
        else:
            bm25 = BM25Retriever.from_texts(all_texts)
            bm25.k = k
            vec_ret = vs.as_retriever(search_kwargs={"k": k})
            retriever = SimpleEnsembleRetriever(
                retrievers=[bm25, vec_ret],
                weights=[0.5, 0.5]
            )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return chain

def get_qa_chain(collection_name="活着", **kwargs):
    safe = resolve_collection_name(collection_name)
    return build_qa_chain(collection_name=safe, **kwargs)

if __name__ == "__main__":
    vs, llm = init_models("活着")
    qa = build_qa_chain(k=7, retrieval_type="ensemble")
    print("\n《活着》知识库问答系统已就绪（输入 'exit' 退出）\n")
    while True:
        query = input("你的问题: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        res = qa.invoke({"query": query})
        print(res["result"])
        print("-" * 50)
