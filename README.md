# 🥑 牛油果 AI 训练助手 (Avocado AI Coach)

一个基于**专属微调 Embedding 模型**与**两阶段 RAG 架构**的私人健身顾问 Agent。采用全栈独立开发，提供极致的流式交互与数据驱动的自学习闭环。

## ✨ 核心特性

*   **🧠 垂直领域微调**：基于 `BAAI/bge-small-zh-v1.5` 在 330 条体育三元组上微调，产出 512 维领域专属模型。纯向量检索 Hit@5 达 0.84（相对基线提升超 50%）。
*   **🔍 两阶段检索架构**：采用 BM25 + RRF 混合粗排召回，结合 `bge-reranker-v2-m3` 精排，将系统 Hit@5 提升至 **0.98**，极大地优化了 RAG 检索精度。
*   **🔄 自学习反馈闭环**：利用用户点踩反馈，自动触发 LLM 生成反思摘要，动态注入 System Prompt，实现 AI 持续进化。
*   **💾 多用户持久化**：FastAPI + SQLite，支持 JWT 鉴权、身体档案持久化与跨会话历史记忆。
*   **🎨 极客前端交互**：原生 JS/CSS 手写深色毛玻璃 UI，包含 Apple 风进场动画、3D 滚轮选择器，支持 SSE 流式打字机渲染。

## ⚠️ Demo 运行说明

由于本项目包含本地微调的 Embedding 模型（`fit-sport-emb`）和交叉编码器重排模型（`bge-reranker-v2-m3`），直接公网部署需要较高的 GPU 显存支持。为了控制云服务成本，目前采用**本地化私有部署**。

如需在线体验，可随时联系我开启内网穿透服务，或支持屏幕共享进行实时运行演示。

## 🛠️ 技术栈

*   **后端**：Python / FastAPI / ChromaDB / SQLite / PyTorch / Sentence-Transformers
*   **模型**：BAAI/bge-small-zh-v1.5 (微调) / BAAI/bge-reranker-v2-m3
*   **前端**：原生 HTML / CSS / JavaScript (SSE 流式渲染)

## 🚀 本地启动指南

1. 克隆仓库并安装依赖：
   ```bash
   git clone https://github.com/avocado0711/avocado-ai-coach.git
   cd avocado-ai-coach
   pip install -r requirements.txt
