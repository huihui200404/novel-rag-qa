import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import itertools

load_dotenv()

# 加载知识库（只需一次）
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./huozhe_chroma", embedding_function=embedding)

# 加载测试集
with open("test_set.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

# 裁判LLM
judge = ChatOpenAI(model="gpt-4o", temperature=0)

# 我们要测试的参数组合
k_values = [3, 5, 7]
# 阈值：注意，Chroma的search函数可以传入similarity_threshold
threshold_values = [0.5, 0.65, 0.8]

# 固定提示词模板
prompt_template = """基于以下上下文回答问题。如果无法从上下文中找到答案，请说“不知道”。

上下文：
{context}

问题：{question}
答案："""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

all_experiments = []

for k, threshold in itertools.product(k_values, threshold_values):
    print(f"\n===== 实验组合: K={k}, 阈值={threshold} =====")
    
    # 为每个组合创建独立的检索器和链
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # 注意：Chroma本身不直接提供score过滤的search方法，通常需要后处理。
    # 我们可以在检索后手动过滤，或者使用vectorstore.similarity_search_with_score
    # 这里展示更精细的控制：检索 + 分数过滤
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    correct = 0
    details = []
    
    for case in test_cases:
        # 获取带分数的文档
        docs_with_scores = vectorstore.similarity_search_with_score(case["question"], k=k)
        # 过滤
        filtered_docs = [doc for doc, score in docs_with_scores if score >= threshold]
        
        if not filtered_docs:
            # 如果全被过滤，回答不知道
            predicted = "不知道"
        else:
            # 手动拼接上下文
            context_text = "\n\n".join([d.page_content for d in filtered_docs])
            # 直接调用LLM，绕过RetrievalQA（因为需要自定义过滤逻辑）
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            prompted_text = PROMPT.format(context=context_text, question=case["question"])
            predicted = llm.invoke(prompted_text).content
        
        # 评判
        judge_prompt = f"""标准答案：{case['ground_truth']}
预测答案：{predicted}
只回复“正确”或“错误”，并说明理由。"""
        eval_res = judge.invoke(judge_prompt).content
        is_correct = "正确" in eval_res
        if is_correct:
            correct += 1
        details.append({
            "id": case["id"],
            "question": case["question"],
            "predicted": predicted,
            "is_correct": is_correct,
            "eval": eval_res
        })
    
    accuracy = correct / len(test_cases) * 100
    experiment = {
        "params": {"k": k, "threshold": threshold},
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "details": details
    }
    all_experiments.append(experiment)
    print(f"准确率: {accuracy:.1f}%")

# 保存所有实验数据
with open("param_sweep_results.json", "w", encoding="utf-8") as f:
    json.dump(all_experiments, f, ensure_ascii=False, indent=2)

# 生成最终对比表
print("\n\n===== 参数调优最终对比表 =====")
print("{:<10} {:<10} {:<10}".format("K", "阈值", "准确率"))
for exp in all_experiments:
    p = exp["params"]
    print("{:<10} {:<10} {:.1f}%".format(p["k"], p["threshold"], exp["accuracy"]))

# 找出最佳组合
best = max(all_experiments, key=lambda x: x["accuracy"])
print(f"\n最佳参数: K={best['params']['k']}, 阈值={best['params']['threshold']}, 准确率={best['accuracy']:.1f}%")