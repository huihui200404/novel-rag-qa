# router.py
import chromadb

# 初始化 Chroma 永久客户端（全局复用）
client = chromadb.PersistentClient(path="./chroma_db")

# 拼音 -> 中文显示名的映射（新增书籍时在这两个地方同时添加）
PINYIN_TO_CHINESE = {
    "huozhe": "活着",
    "xusanguan": "许三观卖血记",
}

def get_all_collections():
    """返回当前 Chroma 中所有 collection 的名称列表（拼音）"""
    return [col.name for col in client.list_collections()]

def get_book_name(pinyin_name: str) -> str:
    """根据拼音返回对应的中文书名，用于界面展示"""
    return PINYIN_TO_CHINESE.get(pinyin_name, pinyin_name)

def detect_book_from_query(query: str):
    """根据问题关键词自动检测用户想查哪本书"""
    query_lower = query.lower()
    for pinyin, chinese in PINYIN_TO_CHINESE.items():
        if chinese in query or pinyin in query_lower:
            return pinyin
    return None  # 无法自动识别时返回 None

def resolve_collection_name(book_name_or_pinyin: str):
    """
    接收书名（中文）或拼音，返回对应的 collection 名称（拼音）。
    如果未指定，返回 None。
    """
    if not book_name_or_pinyin:
        return None
    
    # 重新获取最新集合列表，防止新增书籍后未刷新
    current_collections = get_all_collections()
    
    # 如果是拼音直接返回
    if book_name_or_pinyin in current_collections:
        return book_name_or_pinyin
    
    # 中文名 → 拼音
    for pinyin, chinese in PINYIN_TO_CHINESE.items():
        if chinese == book_name_or_pinyin:
            return pinyin
    
    # 找不到匹配，返回 None
    return None
