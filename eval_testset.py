# eval_testset.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from chat import get_qa_chain

# ================== 测试集 ==================
testset = [
    {"id": 1, "question": "福贵的妻子叫什么名字？", "expected": ["家珍"]},
    {"id": 2, "question": "福贵的儿子叫什么？", "expected": ["有庆"]},
    {"id": 3, "question": "福贵最开始输掉家产是通过什么方式？", "expected": ["赌"]},
    {"id": 4, "question": "福贵一生经历了哪些亲人的去世？", "expected": ["父亲","母亲","家珍","凤霞","二喜","有庆","苦根"]},
    {"id": 5, "question": "有庆是怎么死的？", "expected": ["献血","抽血"]},
    {"id": 6, "question": "家珍得了什么病？", "expected": ["软骨"]},
    {"id": 7, "question": "凤霞为什么会变成聋哑人？", "expected": ["发烧"]},
    {"id": 8, "question": "苦根是怎么死的？", "expected": ["豆子","撑"]},
    {"id": 9, "question": "福贵输掉家产后，他的父亲是如何死的？", "expected": ["粪缸","摔"]},
    {"id": 10, "question": "春生是做什么的？曾和福贵在哪里相遇？", "expected": ["当兵","国民党","战场"]},
    {"id": 11, "question": "凤霞的丈夫叫什么？他的外貌特征是什么？", "expected": ["二喜","偏头"]},
    {"id": 12, "question": "为什么余华让福贵最后只剩一头老牛？", "expected": ["象征","孤独","韧性","陪伴"]},
    {"id": 13, "question": "《活着》的书名有什么反讽意味？", "expected": ["表面活着","苦难","承受"]},
    {"id": 14, "question": "福贵说“我是有时候想想伤心，有时候想想又很踏实”是在什么心境下？", "expected": ["经历死亡","回忆","亲情"]},
    {"id": 15, "question": "小说开头“我”在乡间收集民谣时遇到福贵，这种叙事结构有什么作用？", "expected": ["旁观者","距离","真实"]},
]

def evaluate(qa_chain, testset):
    results = []
    for item in testset:
        query = item["question"]
        try:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "")
        except Exception as e:
            answer = f"ERROR: {e}"
        is_correct = any(keyword in answer for keyword in item["expected"])
        results.append({
            "id": item["id"],
            "question": query,
            "answer_snippet": answer[:200],
            "expected_keywords": item["expected"],
            "correct": is_correct
        })
        print(f"Q{item['id']:02d}: {'✓' if is_correct else '✗'} {query}")
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(testset)
    print(f"\n总准确率: {accuracy:.1%} ({correct_count}/{len(testset)})")
    return results, accuracy

if __name__ == "__main__":
    # 使用你当前最佳配置：混合检索，K=7
    qa = get_qa_chain(k=7, retrieval_type="ensemble")
    results, acc = evaluate(qa, testset)
    with open("eval_baseline.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "details": results}, f, ensure_ascii=False, indent=2)
    print("\n基线结果已保存到 eval_baseline.json")