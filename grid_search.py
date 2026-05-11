from chat import build_qa_chain   # 你需要重构 chat.py，把链构建部分做成函数，接收k、threshold等参数
from eval_testset import testset, evaluate
import json
import itertools

# 参数组合
k_values = [3, 5, 7]
threshold_values = [0.5, 0.65, 0.7]
# 注意：score_threshold 需要检索器支持，你之前代码中使用了 similarity_score_threshold
# 如果你的 ensemble_retriever 不好直接加阈值，可以先单独测试向量检索器的阈值

results_log = []

for k in k_values:
    # 这里只演示调整K，score_threshold 类似处理
    qa_chain = build_qa_chain(k=k, score_threshold=None, retrieval_type="vector")  # 示例
    _, acc = evaluate(qa_chain, testset)
    results_log.append({"k": k, "threshold": "none", "accuracy": acc})
    print(f"K={k}, 准确率={acc:.1%}")

# 保存结果
with open("param_tuning.json", "w", encoding="utf-8") as f:
    json.dump(results_log, f, ensure_ascii=False, indent=2)