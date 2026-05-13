import sys
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 命令行用法: python add_book.py <txt_path> <book_name_chinese>
if len(sys.argv) != 3:
    print("用法: python add_book.py <txt文件路径> <中文书名>")
    sys.exit(1)

txt_path = sys.argv[1]
book_name = sys.argv[2]

# 拼音转换简表（仅用于生成 collection 名，无需完备，手动映射几个字母）
def to_pinyin(name):
    # 简单的硬编码映射，新书你手工加一行
    mapping = {
        "活着": "huozhe",
        "许三观卖血记": "xusanguan",
        "兄弟": "xiongdi",
        # 继续添加...
    }
    return mapping.get(name, name.replace(" ", "").lower())

collection_name = to_pinyin(book_name)
print(f"正在处理: {book_name} -> collection: {collection_name}")

loader = TextLoader(txt_path, encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"分为 {len(chunks)} 个片段")

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db",
    collection_name=collection_name,
    collection_metadata={"chinese_name": book_name}
)
# 新版 Chroma 自动持久化
print(f"✅ 知识库 '{book_name}' 构建成功！collection: {collection_name}")
