from rag_system import get_qa_chain

chain = get_qa_chain(k=5, retrieval_type="vector")
res = chain.invoke({"query": "福贵的儿子叫什么？"})
print("回答：", res["result"])
print("来源数：", len(res.get("source_documents", [])))