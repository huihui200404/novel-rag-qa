# 📚 Novel-RAG-QA：多书籍文学知识库问答系统

基于 **LangChain + Chroma + 智谱 GLM-4** 的企业级 RAG 问答系统，支持《活着》、《许三观卖血记》等多本小说的自动路由、混合检索与原文溯源。

**🌐 在线演示**：[https://novel-rag-qa-production.up.railway.app](https://novel-rag-qa-production.up.railway.app)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ 核心特性

- **多知识库自动路由**：根据问题关键词自动识别查询哪本书（例如“福贵” → 《活着》，“许三观” → 《许三观卖血记》）
- **混合检索（BM25 + 向量）**：兼顾关键词精确匹配与语义模糊搜索，在15题测试集上准确率达 **73.3%**，较纯向量检索提升 6.6%
- **噪音过滤与幻觉抑制**：自动去除“全书完”“版权声明”等噪音片段；强化 Prompt 杜绝编造章节、页码
- **精确缓存**：基于 TTLCache 缓存问答结果，相同问题秒级响应，节省 API 调用费用
- **全链路日志**：记录每次问答的书籍、检索片段数、耗时，便于回溯与优化
- **一键部署**：支持 Docker / Railway 云部署，零配置启动

---

## 🏗️ 系统架构
