# retrieval_eval.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from chat import get_qa_chain
from eval_testset import testset, evaluate

strategies = [
    {"name": "纯向量检索", "type": "vector", "k": 7},
    {"name": "纯BM25检索", "type": "bm25", "k": 7},
    {"name": "混合检索(0.5:0.5)", "type": "ensemble", "k": 7}
]

results_log = []
for strat in strategies:
    print(f"\n正在评估：{strat['name']} ...")
    qa = get_qa_chain(k=strat["k"], retrieval_type=strat["type"])
    res, acc = evaluate(qa, testset)
    results_log.append({
        "strategy": strat["name"],
        "accuracy": acc
    })
    print(f"准确率: {acc:.1%}")

# 打印对比
print("\n检索策略准确率对比：")
for r in results_log:
    print(f"  {r['strategy']}: {r['accuracy']:.1%}")

with open("retrieval_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results_log, f, ensure_ascii=False, indent=2)