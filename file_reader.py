# 文件读取 + 简单分段演示
FILE_PATH = "data/huozhe.txt"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

print(f"小说《活着》共加载 {len(content)} 个字符")
print("===== 前 200 字预览 =====")
print(content[:200])

# 按句号粗暴分段，感受下数据形态
sentences = content.split("。")
print(f"\n按句号分割，共 {len(sentences)} 个句子")
print("前 3 个句子：")
for i, sent in enumerate(sentences[:3], 1):
    print(f"  {i}. {sent.strip()}。")