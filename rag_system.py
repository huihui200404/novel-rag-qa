# rag_system.py
import os
# 以下两行已注释，允许 Railway 从 Hugging Face 下载模型
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

# 依赖 router.py 中的动态路由函数
from router import (
    resolve_collection_name,      # 中文/拼音 → 拼音 collection 名
    detect_book_from_query,       # 从问题中自动识别书名
    get_all_collections,          # 获取所有已构建的 collection 名
    get_book_name,                # 拼音 → 中文显示名
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from typing import List, Optional
from langchain_core.documents import Document

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

def init_models(collection_name: Optional[str] = None):
    """
    初始化 Embedding 模型、向量库和 LLM。
    如果 collection_name 为 None，则自动选择第一个可用的集合。
    """
    global _vectorstore, _llm, _embedding_model, _current_collection

    if collection_name:
        # 将可能的书名或拼音转换为 collection 名
        collection_name = resolve_collection_name(collection_name)
    else:
        # 未指定时，获取第一个可用的 collection
        available = get_all_collections()
        if not available:
            raise RuntimeError("未找到任何向量库，请先使用 add_book.py 构建知识库。")
        collection_name = available[0]
        print(f"未指定书籍，自动选择: {collection_name}")

    # 如果向量库已经加载且目标没变，直接返回
    if _vectorstore is not None and _llm is not None and _current_collection == collection_name:
        return _vectorstore, _llm

    # 加载 Embedding 模型（全局共享）
    if _embedding_model is None:
        print("正在加载 Embedding 模型...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    # 加载指定 collection 的向量库
    print(f"正在加载向量库: {collection_name}")
    _vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=_embedding_model,
        collection_name=collection_name
    )
    _current_collection = collection_name
    doc_count = len(_vectorstore.get()["documents"])
    print(f"向量库 [{collection_name}] 包含 {doc_count} 个文档片段")

    # 加载 LLM（全局共享）
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


def build_qa_chain(collection_name: Optional[str] = None, k: int = 7, retrieval_type: str = "ensemble"):
    """
    构建完整的 RAG 链。
    collection_name: 可以是中文书名或拼音（如 "活着" 或 "huozhe"），为 None 时自动选。
    k: 检索返回的文档数量
    retrieval_type: "vector", "bm25", "ensemble"
    """
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


def get_qa_chain(book: Optional[str] = None, k: int = 7, retrieval_type: str = "ensemble"):
    """
    便捷接口，供 Chainlit 或外部调用。
    book: 书籍名称（中文/拼音均可），None 则自动选择第一个可用的。
    """
    if book:
        book = resolve_collection_name(book)  # 确保统一为拼音
    return build_qa_chain(collection_name=book, k=k, retrieval_type=retrieval_type)


def get_available_books():
    """
    返回所有已构建知识库的书籍中文名列表，用于界面提示。
    """
    collections = get_all_collections()
    return [get_book_name(c) for c in collections]


# ---------- 本地测试 ----------
if __name__ == "__main__":
    print("可用的书籍:", get_available_books())
    qa = get_qa_chain()  # 自动选第一本
    print("\n文学知识库问答系统已就绪（输入 'exit' 退出）\n")
    while True:
        query = input("你的问题: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        res = qa.invoke({"query": query})
        print(res["result"])
        print("-" * 50)
