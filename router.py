import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_db")

# 自动获取所有 collection 名字（拼音）
ALL_COLLECTIONS = [col.name for col in client.list_collections()]

# 拼音 -> 中文显示名的映射，可以从 collection 的 metadata 读，但我们先用配置文件
# 这里做一个简单映射表，可以手动维护，或者用 json 配置文件
PINYIN_TO_CHINESE = {
    "huozhe": "活着",
    "xusanguan": "许三观卖血记",
    # 新增书籍在这两处同时添加
}

def detect_book_from_query(query: str):
    """根据问题关键词自动检测用户想查哪本书"""
    query_lower = query.lower()
    # 遍历所有 collection，检查中文名或拼音是否出现在提问中
    for pinyin, chinese in PINYIN_TO_CHINESE.items():
        if chinese in query or pinyin in query_lower:
            return pinyin
    return None  # 无法自动识别，退回手动选择或默认

def resolve_collection_name(book_name_or_pinyin: str):
    """接收书名或拼音，返回 collection 名称，如果未指定则返回 None 触发多选"""
    if not book_name_or_pinyin:
        return None
    # 如果是拼音直接返回
    if book_name_or_pinyin in ALL_COLLECTIONS:
        return book_name_or_pinyin
    # 如果是中文名，找对应拼音
    for pinyin, chinese in PINYIN_TO_CHINESE.items():
        if chinese == book_name_or_pinyin:
            return pinyin
    return None
