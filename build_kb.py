# build_kb.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHUNK_SIZE = 500
OVERLAP = 50

def build_vectorstore(file_path, book_name):
    """
    加载小说文本，分段，向量化并存储到 Chroma。
    book_name 必须是英文/拼音（如 huozhe, xusanguan），会作为 collection 名。
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

    # 3. 向量化并存储（Chroma 会自动持久化，无需 .persist()）
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db",
        collection_name=book_name
    )
    return len(chunks)