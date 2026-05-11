# app.py
import os
import chainlit as cl
from rag_system import get_qa_chain
from build_kb import build_vectorstore
from router import get_book_name
from cache_manager import CacheManager
from loguru import logger
import traceback

# ---------- 日志配置 ----------
logger.add(
    "rag_qa.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# ---------- 全局缓存（TTL 1小时，最多100条） ----------
cache = CacheManager(maxsize=100, ttl=3600)


@cl.on_chat_start
async def start():
    """首次加载：初始化默认知识库（活着）"""
    try:
        chain = get_qa_chain(collection_name="活着", k=7, retrieval_type="ensemble")
        cl.user_session.set("chain", chain)
        cl.user_session.set("current_book", "活着")
        await cl.Message(
            content="你好！我是文学知识库问答助手。\n"
                    "你可以直接提问（系统会自动识别书籍），或上传 .txt 文件并输入拼音名来创建新知识库。"
        ).send()
        logger.info("系统初始化成功，默认知识库：活着")
    except Exception as e:
        await cl.Message(content=f"❌ 系统初始化失败：{str(e)}").send()
        logger.error(f"初始化失败：{traceback.format_exc()}")


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()

    # ========== 1. 上传文件 → 构建新知识库 ==========
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
                        content="📌 请在消息框输入拼音库名（如 huozhe 或 xusanguan）。"
                    ).send()
                    return

                # 进度提示
                progress = cl.Message(content=f"⏳ 正在构建《{book_name}》知识库...")
                await progress.send()

                try:
                    chunk_count = build_vectorstore(file_path, book_name)
                    # 更新当前链和会话
                    new_chain = get_qa_chain(collection_name=book_name, k=7, retrieval_type="ensemble")
                    cl.user_session.set("chain", new_chain)
                    cl.user_session.set("current_book", book_name)
                    progress.content = f"✅ 知识库《{book_name}》构建完成！共 {chunk_count} 个片段。"
                    logger.info(f"知识库构建成功：{book_name}，片段数：{chunk_count}")
                except Exception as e:
                    progress.content = f"❌ 构建失败：{str(e)}"
                    logger.error(f"构建失败：{traceback.format_exc()}")
                await progress.update()
                return  # 上传完成，不进行问答

    # ========== 2. 正常问答流程 ==========
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="🔧 系统尚未就绪，请刷新页面重试。").send()
        return

    if not user_input:
        return

    # ---------- 2.1 意图识别（自动路由） ----------
    recognized_book = get_book_name(user_input)
    current_book = cl.user_session.get("current_book", "活着")
    if recognized_book != "其他" and recognized_book != current_book:
        try:
            new_chain = get_qa_chain(collection_name=recognized_book, k=7, retrieval_type="ensemble")
            cl.user_session.set("chain", new_chain)
            cl.user_session.set("current_book", recognized_book)
            chain = new_chain
            logger.info(f"路由切换至：{recognized_book}")
        except Exception:
            logger.error(f"路由切换失败：{traceback.format_exc()}")

    # ---------- 2.2 精确缓存 ----------
    cached = cache.get(user_input)
    if cached:
        logger.info(f"缓存命中：{user_input[:50]}...")
        await cl.Message(content=cached["answer"]).send()
        return

    # ---------- 2.3 执行 RAG 链（异步调用） ----------
    logger.info(f"用户提问：{user_input}")
    try:
        # 如果异步调用报错，可替换为 res = chain.invoke({"query": user_input})
        res = await chain.ainvoke({"query": user_input})
        answer = res["result"]
        source_docs = res.get("source_documents", [])
        logger.info(f"回答生成成功，检索到 {len(source_docs)} 个片段")
    except Exception as e:
        answer = f"❌ 处理出错：{str(e)}"
        source_docs = []
        logger.error(f"处理失败：{traceback.format_exc()}")

    # ---------- 2.4 构建回复消息（含引用） ----------
    refs = []
    for i, doc in enumerate(source_docs[:3]):
        snippet = doc.page_content[:200].replace("\n", " ")
        refs.append(f"📖 片段{i+1}：{snippet}...")
    ref_text = "\n\n".join(refs)

    final_msg = f"{answer}\n\n---\n{ref_text}" if ref_text else answer

    # 存入缓存
    cache.set(user_input, {"answer": final_msg})
    await cl.Message(content=final_msg).send()