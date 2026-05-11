# eval_rerank.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from chat import get_qa_chain
from eval_testset import testset, evaluate
from rerank_chain import qa_rerank  # 导入刚才构建的 rerank 链

print("评估混合检索（无 Rerank）...")
qa_ensemble = get_qa_chain(k=7, retrieval_type="ensemble")
res_ensemble, acc_ensemble = evaluate(qa_ensemble, testset)
print(f"混合检索准确率: {acc_ensemble:.1%}")

print("\n评估混合检索 + Rerank ...")
res_rerank, acc_rerank = evaluate(qa_rerank, testset)
print(f"+ Rerank 准确率: {acc_rerank:.1%}")

print(f"\n准确率提升: {acc_rerank - acc_ensemble:.1%}")

with open("rerank_eval.json", "w", encoding="utf-8") as f:
    json.dump({
        "ensemble_accuracy": acc_ensemble,
        "rerank_accuracy": acc_rerank,
        "improvement": acc_rerank - acc_ensemble
    }, f, indent=2)