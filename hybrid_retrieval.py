from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 加载向量存储
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./huozhe_chroma", embedding_function=embedding)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. 构建BM25关键词检索器
# 需要先拿到所有documents（可以从vectorstore获取，或直接从源文档加载）
all_docs = vectorstore.get()  # 可能不直接支持，简单起见，我们从阶段2的文档列表加载
# 如果你在build_vectorstore.py里保存了documents列表，可以直接用。
# 这里演示：重新分割一次（仅示例，实际最好缓存）
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("huozhe.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 5

# 3. 创建混合检索器，权重各0.5
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# 4. 测试一个例子
docs = ensemble_retriever.invoke("有庆是怎么死的？")
for i, doc in enumerate(docs):
    print(f"混合检索结果 {i+1}: {doc.page_content[:100]}...")