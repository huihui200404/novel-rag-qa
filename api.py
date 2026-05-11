from fastapi import FastAPI
from pydantic import BaseModel
from rag_system import get_qa_chain
from router import get_book_name
from cache_manager import CacheManager

app = FastAPI()
cache = CacheManager()

class AskRequest(BaseModel):
    question: str
    book: str = None

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    book = req.book or get_book_name(req.question)
    if book == "其他": book = "活着"
    cached = cache.get(req.question)
    if cached:
        return AskResponse(answer=cached["answer"], sources=[])
    chain = get_qa_chain(collection_name=book, k=7, retrieval_type="ensemble")
    res = chain.invoke({"query": req.question})
    sources = [d.page_content[:200] for d in res.get("source_documents", [])[:3]]
    ans = res["result"]
    cache.set(req.question, {"answer": ans})
    return AskResponse(answer=ans, sources=sources)