from sentence_transformers import CrossEncoder
model = CrossEncoder(
    "D:/你的路径/models--BAAI--bge-reranker-base/snapshots/xxx",
    local_files_only=True
)
print("加载成功")