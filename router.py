# router.py
def get_book_name(question: str) -> str:
    """根据问题中的关键词判断用户想查询哪本书，返回书名（中文）"""
    q = question.lower()
    # 活着 特征词
    if any(w in q for w in ["福贵", "有庆", "家珍", "凤霞", "二喜", "苦根", "老牛", "徐有庆"]):
        return "活着"
    # 许三观卖血记 特征词
    if any(w in q for w in ["许三观", "卖血", "一乐", "二乐", "三乐", "许玉兰", "林芬芳"]):
        return "许三观卖血记"
    # 无法判断时返回 "其他"
    return "其他"