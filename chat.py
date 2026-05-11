# chat.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"          # 强制离线，不再连接 HuggingFace
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 仅首次下载时需要，可以保留无害

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import langchain_openai
from langchain_community.retrievers import BM25Retriever
import langchain.retrievers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ========== 提示词（可继续从 step4_prompt 导入，这里内联以便独立运行） ==========
TEMPLATE = """# 角色
你是一位中国现当代文学研究员，专攻余华作品。

# 任务
基于【知识库】中的小说原文，回答用户关于《活着》的问题。

# 输出格式（严格遵守）
1. **核心结论**：一句话概括答案。
2. **文本依据**：引用至少两处原文片段（注明大致章节）。
3. **深层解读**：结合人物命运或时代背景，扩展说明。

# 边界规则
- 如果知识库中没有任何相关信息，回复：“根据现有资料，无法确认该问题。请尝试询问具体情节或人物。”
- 严禁使用外部知识或主观猜测。

# 上下文
{context}

# 用户问题
{question}
"""
QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=TEMPLATE)

# ========== 全局缓存（避免重复加载） ==========
_vectorstore = None
_llm = None
_embedding_model = None

def init_models():
    """初始化 Embedding、向量库和 LLM（仅首次运行时执行）"""
    global _vectorstore, _llm, _embedding_model
    if _vectorstore is not None and _llm is not None:
        return _vectorstore, _llm

    print("正在加载 Embedding 模型...")
    _embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("正在加载向量库...")
    _vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=_embedding_model
    )

    print("正在初始化智谱 LLM...")
    _llm = langchain_openai.ChatOpenAI(
        model="glm-4-flash",
        temperature=0,
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        max_tokens=2048
    )
    return _vectorstore, _llm


def build_qa_chain(k=5, score_threshold=None, retrieval_type="ensemble"):
    """
    构建可定制的检索增强问答链
    
    参数:
        k: 检索返回的文档数量
        score_threshold: 向量检索的分数阈值（None 表示不过滤）
        retrieval_type: "vector" 纯向量, "bm25" 纯BM25, "ensemble" 混合（默认）
    返回:
        RetrievalQA 链对象
    """
    # 确保模型已加载
    vs, llm = init_models()

    # 根据 retrieval_type 构建检索器
    if retrieval_type == "vector":
        retriever = vs.as_retriever(
            search_kwargs={"k": k, "score_threshold": score_threshold} if score_threshold else {"k": k}
        )
    elif retrieval_type == "bm25":
        all_texts = vs.get()["documents"]
        bm25 = BM25Retriever.from_texts(all_texts)
        bm25.k = k
        retriever = bm25
    else:  # ensemble
        all_texts = vs.get()["documents"]
        bm25 = BM25Retriever.from_texts(all_texts)
        bm25.k = k
        vec_ret = vs.as_retriever(
            search_kwargs={"k": k, "score_threshold": score_threshold} if score_threshold else {"k": k}
        )
        retriever = langchain.retrievers.EnsembleRetriever(
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


def get_qa_chain(**kwargs):
    """外部调用接口：直接返回可 invoke 的链对象"""
    return build_qa_chain(**kwargs)


# ========== 原有的交互式主程序 ==========
if __name__ == "__main__":
    # 初始化模型
    vs, llm = init_models()

    # 构建默认混合检索器（K=7，你之前调优的值）
    qa = build_qa_chain(k=7, retrieval_type="ensemble")

    print("\n《活着》知识库问答系统已就绪（输入 'exit' 退出）\n")

    SHOW_SCORES = True  # 是否打印检索分数，改为 False 关闭

    while True:
        query = input("你的问题: ").strip()
        if query.lower() == "exit":
            print("再见！")
            break
        if not query:
            continue

        if SHOW_SCORES:
            print("\n--- 检索分数 (Top-5) ---")
            docs_with_scores = vs.similarity_search_with_relevance_scores(query, k=5)
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                print(f"[{score:.3f}] {doc.page_content[:80]}...")
            print("-" * 24)

        try:
            result = qa.invoke({"query": query})
            print(result["result"])
            print("-" * 50)
        except Exception as e:
            print(f"系统出错：{e}")