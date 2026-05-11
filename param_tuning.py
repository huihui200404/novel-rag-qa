# param_tuning.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from chat import get_qa_chain
from eval_testset import testset, evaluate  # 复用你的测试集和评估函数

k_values = [3, 5, 7]
thresholds = [None, 0.5, 0.65]  # None 表示不使用阈值

results_log = []

for k in k_values:
    for th in thresholds:
        print(f"\n{'='*40}")
        print(f"测试组合: K={k}, threshold={th}")
        # 注意：score_threshold 只对向量检索有效，这里我们在纯向量模式下测试
        qa = get_qa_chain(k=k, score_threshold=th, retrieval_type="vector")
        res, acc = evaluate(qa, testset)
        results_log.append({
            "k": k,
            "threshold": str(th),
            "accuracy": acc
        })
        print(f"准确率: {acc:.1%}")

# 找出最佳组合
best = max(results_log, key=lambda x: x["accuracy"])
print("\n" + "="*40)
print("参数调优完成！最佳组合：")
print(f"  K = {best['k']}, threshold = {best['threshold']}")
print(f"  准确率 = {best['accuracy']:.1%}")

# 保存完整结果
with open("param_tuning_results.json", "w", encoding="utf-8") as f:
    json.dump(results_log, f, ensure_ascii=False, indent=2)
print("\n详细结果已保存到 param_tuning_results.json")