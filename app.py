# app.py
import os
import chainlit as cl
from rag_system import get_qa_chain
from build_kb import build_vectorstore
from router import get_book_name
from cache_manager import CacheManager
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_qa.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局缓存
cache = CacheManager(maxsize=100, ttl=3600)


@cl.on_chat_start
async def start():
    """初始化默认知识库（活着）"""
    try:
        chain = get_qa_chain(collection_name="活着", k=7, retrieval_type="ensemble")
        cl.user_session.set("chain", chain)
        cl.user_session.set("current_book", "活着")

        await cl.Message(
            content=(
                "你好，我是文渊。\n\n"
                "这里收录了中国现当代小说的文本细读资料。你可以直接向我提问，"
                "我会从已收录的作品中寻找答案；如果你想添加新的书籍，"
                "也可以上传 .txt 文件，并在消息框中输入书籍的拼音名称。"
            ),
            author="文渊"
        ).send()

    except Exception as e:
        await cl.Message(
            content=f"系统初始化时遇到了一些问题：{str(e)}",
            author="文渊"
        ).send()
        logger.error(traceback.format_exc())


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()

    # ========== 1. 处理文件上传 ==========
    if message.elements:
        for element in message.elements:
            if element.mime == "text/plain":
                file_path = element.path
                if not os.path.exists(file_path):
                    await cl.Message(
                        content="文件读取失败了，请重新上传试试。",
                        author="文渊"
                    ).send()
                    return

                book_name = user_input
                if not book_name:
                    await cl.Message(
                        content="请在上传文件的同时，在消息框中输入书籍的拼音名称，例如 *huozhe* 或 *xusanguan*。",
                        author="文渊"
                    ).send()
                    return

                progress = cl.Message(
                    content=f"正在为你整理《{book_name}》的文本资料，请稍候片刻……",
                    author="文渊"
                )
                await progress.send()

                try:
                    chunk_count = build_vectorstore(file_path, book_name)
                    new_chain = get_qa_chain(
                        collection_name=book_name, k=7, retrieval_type="ensemble"
                    )
                    cl.user_session.set("chain", new_chain)
                    cl.user_session.set("current_book", book_name)

                    progress.content = (
                        f"《{book_name}》已收录完成，共整理出 {chunk_count} 个文本片段，"
                        "现在可以开始提问了。"
                    )
                    logger.info("知识库构建成功: %s, 片段数: %d", book_name, chunk_count)

                except Exception as e:
                    progress.content = (
                        f"很抱歉，整理《{book_name}》时遇到了一些问题：{str(e)}\n\n"
                        "请检查文件格式是否正确，或换一本书名重试。"
                    )
                    logger.error(traceback.format_exc())

                await progress.update()
                return

    # ========== 2. 正常问答 ==========
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(
            content="系统似乎还未准备好，请刷新页面后再试。",
            author="文渊"
        ).send()
        return

    if not user_input:
        return

    # 意图识别（自动路由）
    recognized = get_book_name(user_input)
    current_book = cl.user_session.get("current_book", "活着")

    if recognized != "其他" and recognized != current_book:
        try:
            new_chain = get_qa_chain(
                collection_name=recognized, k=7, retrieval_type="ensemble"
            )
            cl.user_session.set("chain", new_chain)
            cl.user_session.set("current_book", recognized)
            chain = new_chain
            logger.info("路由切换至: %s", recognized)
        except Exception:
            logger.error("路由切换失败: %s", traceback.format_exc())

    # 缓存命中
    cached = cache.get(user_input)
    if cached:
        logger.info("缓存命中: %s", user_input)
        await cl.Message(content=cached["answer"], author="文渊").send()
        return

    # 调用 RAG
    logger.info("用户提问: %s", user_input)
    try:
        res = chain.invoke({"query": user_input})
        answer = res["result"]
        source_docs = res.get("source_documents", [])
        logger.info("检索到 %d 个相关片段", len(source_docs))
    except Exception as e:
        answer = (
            f"抱歉，这个问题我暂时没能处理好：{str(e)}\n\n"
            "你可以换一种方式提问，或者稍后再试。"
        )
        source_docs = []
        logger.error(traceback.format_exc())

    # 构建引用（用 markdown 引用块，更像学术注释）
    refs = []
    for i, doc in enumerate(source_docs[:3]):
        snippet = doc.page_content[:200].replace("\n", " ")
        refs.append(f"> **片段 {i + 1}**｜{snippet}…")

    ref_text = "\n\n".join(refs)

    if ref_text:
        final_msg = f"{answer}\n\n---\n\n**参考文本**\n\n{ref_text}"
    else:
        final_msg = answer

    cache.set(user_input, {"answer": final_msg})
    await cl.Message(content=final_msg, author="文渊").send()