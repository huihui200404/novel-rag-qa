import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("huozhe.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"共切出 {len(chunks)} 个片段")

# 观察前两个片段的重叠部分
print("\n--- 片段 0 的结尾 ---")
print(chunks[0].page_content[-80:])
print("\n--- 片段 1 的开头 ---")
print(chunks[1].page_content[:80])