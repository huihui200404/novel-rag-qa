import json
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# 1. 加载已有知识库
embedding = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./huozhe_chroma",
    embedding_function=embedding
)

# 2. 构建RAG链（使用你阶段2优化好的参数，这里先写基础版）
prompt_template = """基于以下上下文回答问题。如果无法从上下文中找到答案，请说“不知道”。

上下文：
{context}

问题：{question}
答案："""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # temperature=0保证可复现

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT}
)

# 3. 定义评估裁判
judge_llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # 更聪明的裁判

def evaluate_answer(question, predicted_answer, ground_truth):
    """让LLM评估预测答案是否正确"""
    judge_prompt = """你是一个严格的评估员。根据以下标准答案，判断给出的回答是否正确。
    只需回复"正确"或"错误"，并附上简短理由。

    标准答案：{ground_truth}
    预测答案：{predicted_answer}
    
    评估结果："""
    
    response = judge_llm.invoke(judge_prompt.format(
        ground_truth=ground_truth,
        predicted_answer=predicted_answer
    ))
    return response.content

# 4. 批量测试主流程
with open("test_set.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

results = []
correct_count = 0

for item in test_data:
    print(f"正在测试 [{item['type']}] 问题 {item['id']}: {item['question']}")
    
    # RAG问答
    answer = qa_chain.invoke(item["question"])
    predicted = answer["result"]
    
    # LLM评估
    eval_result = evaluate_answer(item["question"], predicted, item["ground_truth"])
    
    is_correct = "正确" in eval_result
    if is_correct:
        correct_count += 1
        
    # 记录
    results.append({
        "id": item["id"],
        "question": item["question"],
        "predicted": predicted,
        "ground_truth": item["ground_truth"],
        "eval_result": eval_result,
        "is_correct": is_correct,
        "type": item["type"]
    })
    print(f"    结果: {'✅正确' if is_correct else '❌错误'} - {eval_result}")

# 5. 计算并保存指标
accuracy = correct_count / len(test_data) * 100
summary = {
    "total": len(test_data),
    "correct": correct_count,
    "accuracy": f"{accuracy:.1f}%",
    "details": results
}

with open("evaluation_results_base.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n测试完成！准确率：{accuracy:.1f}%")
print("详细结果已保存至 evaluation_results_base.json")