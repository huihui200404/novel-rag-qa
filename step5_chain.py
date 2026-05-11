import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings   # 新
from langchain_chroma import Chroma                       # 新
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from step4_prompt import QA_PROMPT

print("1. 加载 Embedding...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("2. Embedding OK")

print("3. 加载向量库...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
print("4. 向量库 OK")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("5. 初始化智谱 LLM...")
llm = ChatOpenAI(
    model="glm-4-flash",
    temperature=0,
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

print("6. 开始调用 LLM...")
query = "福贵是一个怎样的人？"
try:
    result = qa.invoke({"query": query})
    print("\n===== 回答 =====")
    print(result["result"])
    print("\n===== 引用来源（部分）=====")
    for i, doc in enumerate(result["source_documents"]):
        print(f"{i+1}. {doc.page_content[:100]}...")
except Exception as e:
    print(f"LLM 调用失败: {e}")