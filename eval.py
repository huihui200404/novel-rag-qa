import json, time, os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# 使用本地模型，永不联网
local_model_path = os.path.join(os.getcwd(), "local_model", "bge-small-zh-v1.5")
embedding = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

SYSTEM_PROMPT = """# 角色
你是一位中国现当代文学研究员，专攻余华作品。

# 任务
基于以下小说原文片段，回答用户问题。

# 输出格式（严格遵守）
1. **核心结论**：用不少于50字的一句话概括。
2. **文本依据**：引用至少两处原文片段，注明大致章节。
3. **深层解读**：不少于200字，结合人物命运、时代背景、叙事视角展开。

# 边界规则
- 如果片段无法支持回答，回复：“根据现有资料，无法确认该问题。”
- 严禁使用外部知识或主观猜测。

# 上下文
{context}

# 用户问题
{question}"""

QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

def get_retriever():
    persist_dir = "./chroma_all" if os.path.exists("./chroma_all") else "./chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name="huozhe"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 7})

def get_llm():
    return ChatOpenAI(
        model="glm-4-flash",
        temperature=0,
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        max_tokens=2048,
    )

def keyword_hit(answer: str, keywords: list) -> bool:
    """检查答案中是否包含至少一个期望关键词"""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)

def compute_mrr(retrieved_docs, expected_keywords):
    """计算 MRR：第一个包含任一关键词的文档排名的倒数"""
    for rank, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content.lower()
        if any(kw.lower() in content for kw in expected_keywords):
            return 1.0 / rank
    return 0.0

def evaluate(dataset_path="eval_dataset.json", output_path="eval_results.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    retriever = get_retriever()
    llm = get_llm()

    results = []
    total = len(testset)
    hit_count = 0
    mrr_sum = 0.0

    print(f"开始评估 {total} 个问题...\n")
    for item in testset:
        qid = item["id"]
        question = item["question"]
        keywords = item["keywords"]
        print(f"[{qid}] {question}")

        # 检索
        docs = retriever.invoke(question)
        # 生成
        context = "\n\n".join([d.page_content for d in docs])
        prompt = QA_PROMPT.format(context=context, question=question)
        response = llm.invoke(prompt)
        answer = response.content

        # 评分
        hit = keyword_hit(answer, keywords)
        mrr = compute_mrr(docs, keywords)

        if hit:
            hit_count += 1
        mrr_sum += mrr

        results.append({
            "id": qid,
            "difficulty": item["difficulty"],
            "question": question,
            "keywords": keywords,
            "answer_snippet": answer[:200] + "...",
            "keyword_hit": hit,
            "mrr": round(mrr, 4),
            "retrieved_docs_count": len(docs)
        })

        print(f"  → 关键词命中: {hit}, MRR: {mrr:.4f}\n")

    accuracy = hit_count / total
    avg_mrr = mrr_sum / total

    summary = {
        "total_questions": total,
        "accuracy": accuracy,
        "average_mrr": avg_mrr,
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    report = {
        "summary": summary,
        "details": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("========== 评估完成 ==========")
    print(f"准确率 (Keyword Match): {accuracy:.2%} ({hit_count}/{total})")
    print(f"平均 MRR: {avg_mrr:.4f}")
    print(f"详细结果已保存到 {output_path}")

if __name__ == "__main__":
    evaluate()