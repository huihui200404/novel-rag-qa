# app.py
import os
import chainlit as cl
from rag_system import get_qa_chain
from build_kb import build_vectorstore
from router import get_book_name
from cache_manager import CacheManager
import logging
import traceback

# 配置日志（记录到 rag_qa.log，同时输出到终端）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_qa.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局缓存对象
cache = CacheManager(maxsize=100, ttl=3600)

@cl.on_chat_start
async def start():
    """初始化默认知识库（活着）"""
    try:
        chain = get_qa_chain(collection_name="活着", k=7, retrieval_type="ensemble")
        cl.user_session.set("chain", chain)
        cl.user_session.set("current_book", "活着")
        await cl.Message(
            content="你好！我是文学知识库问答助手。\n"
                    "你可以直接提问（系统会自动识别书籍），或上传 .txt 文件并输入拼音名来创建新知识库。"
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ 系统初始化失败：{str(e)}").send()
        logger.error(traceback.format_exc())

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()

    # ========== 1. 处理文件上传（构建新知识库） ==========
    if message.elements:
        for element in message.elements:
            if element.mime == "text/plain":
                file_path = element.path
                if not os.path.exists(file_path):
                    await cl.Message(content="❌ 上传文件读取失败，请重试。").send()
                    return

                book_name = user_input
                if not book_name:
                    await cl.Message(
                        content="📌 请在消息框输入拼音库名（例如 huozhe 或 xusanguan）。"
                    ).send()
                    return

                # 进度提示
                progress = cl.Message(content=f"⏳ 正在构建《{book_name}》知识库，请稍候...")
                await progress.send()

                try:
                    chunk_count = build_vectorstore(file_path, book_name)
                    new_chain = get_qa_chain(collection_name=book_name, k=7, retrieval_type="ensemble")
                    cl.user_session.set("chain", new_chain)
                    cl.user_session.set("current_book", book_name)
                    progress.content = f"✅ 知识库《{book_name}》构建完成！共 {chunk_count} 个片段。"
                    logger.info("知识库构建成功: %s, 片段数: %d", book_name, chunk_count)
                except Exception as e:
                    progress.content = f"❌ 构建失败：{str(e)}"
                    logger.error(traceback.format_exc())
                await progress.update()
                return  # 上传完成，不继续问答

    # ========== 2. 正常问答 ==========
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="🔧 系统尚未就绪，请刷新页面重试。").send()
        return

    if not user_input:
        return

    # ---------- 2.1 意图识别（自动路由） ----------
    recognized = get_book_name(user_input)
    current_book = cl.user_session.get("current_book", "活着")
    if recognized != "其他" and recognized != current_book:
        try:
            new_chain = get_qa_chain(collection_name=recognized, k=7, retrieval_type="ensemble")
            cl.user_session.set("chain", new_chain)
            cl.user_session.set("current_book", recognized)
            chain = new_chain
            logger.info("路由切换至: %s", recognized)
        except Exception:
            logger.error("路由切换失败: %s", traceback.format_exc())

    # ---------- 2.2 精确缓存 ----------
    cached = cache.get(user_input)
    if cached:
        logger.info("缓存命中: %s", user_input)
        await cl.Message(content=cached["answer"]).send()
        return

    # ---------- 2.3 调用 RAG 链 ----------
    logger.info("用户提问: %s", user_input)
    try:
        res = chain.invoke({"query": user_input})
        answer = res["result"]
        source_docs = res.get("source_documents", [])
        logger.info("检索到 %d 个相关片段", len(source_docs))
    except Exception as e:
        answer = f"❌ 处理出错：{str(e)}"
        source_docs = []
        logger.error(traceback.format_exc())

    # ---------- 2.4 构建回答（含引用） ----------
    refs = []
    for i, doc in enumerate(source_docs[:3]):
        snippet = doc.page_content[:200].replace("\n", " ")
        refs.append(f"📖 片段{i+1}：{snippet}...")
    ref_text = "\n\n".join(refs)

    final_msg = f"{answer}\n\n---\n{ref_text}" if ref_text else answer

    # 存入缓存
    cache.set(user_input, {"answer": final_msg})
    await cl.Message(content=final_msg).send()