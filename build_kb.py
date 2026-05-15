# build_kb.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHUNK_SIZE = 500
OVERLAP = 50


def _get_embedding_model():
    """
    自适应获取 Embedding 模型：
    - 如果环境变量 MODEL_PATH 存在且指向有效目录 → 使用该本地路径，强制离线
    - 否则 → 从 HuggingFace 在线加载（适合本地开发）
    """
    model_path = os.environ.get("MODEL_PATH", "")
    if model_path and os.path.isdir(model_path):
        # Docker 环境：模型已预装，离线读取
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"✅ 使用本地 Embedding 模型：{model_path}")
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # 本地开发：允许联网或从缓存加载
        os.environ.pop("HF_HUB_OFFLINE", None)
        print("ℹ️ 本地模型未设置，将从 HuggingFace 加载 BAAI/bge-small-zh-v1.5")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


def build_vectorstore(file_path, book_name):
    """
    加载小说文本，分段，向量化并存储到 Chroma。
    book_name 必须是英文/拼音，会作为 collection 名。
    返回生成的文本片段总数。
    """
    # 1. 加载文本
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # 2. 分割成小块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["book"] = book_name

    # 3. 获取 Embedding 模型（优先本地）
    embedding = _get_embedding_model()

    # 4. 向量化并存储（Chroma 自动持久化）
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db",
        collection_name=book_name
    )
    return len(chunks)