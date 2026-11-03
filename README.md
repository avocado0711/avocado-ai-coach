# Avocado AI Coach

面向力量训练场景的本地 RAG 健身助手。项目不仅提供问答，还把用户档案、
训练计划、训练日志、周复盘和反馈审核串成一条可执行的闭环。

## 目录结构

```text
app/                                            # FastAPI 与 RAG 运行时核心
static/                                         # 前端页面
docs/                                           # 知识文本与审核约定
evaluation/                                     # 检索、生成、Embedding 数据集
scripts/                                        # 重建、评测、备份、报表工具
tests/                                          # Python 与静态检查
```

## 核心能力

- 用户注册、JWT 登录、身体档案和历史会话持久化。
- BM25 + 向量召回 + RRF 融合 + BGE 重排的唯一检索链路，线上和离线共用。
- 真实相关性分数、阈值拒答和无关问题边界，不再给任意问题强行拼接知识库。
- 红旗症状、医疗诊断请求、提示词注入和高风险训练请求的确定性安全拦截。
- 流式回答、知识来源提示、结构化训练计划、训练记录和周复盘。
- 下一版计划读取上一版的完成率、疼痛、精力和 RPE，记录父计划、版本号和调整原因。
- 点赞/点踩与点踩原因；反馈默认隔离，管理员审核后才可能进入后续提示。
- 账号数据导出、账号删除、隐私说明、聊天限流和响应安全头。

AI 输出不构成医疗诊断或治疗建议。正式公开试点前仍需由合格教练或康复专业人员
审核知识内容和安全规则。

本地 RAG 只表示知识片段存储在本地。生成回答仍会把当前对话、必要的身体档案和
检索片段发送给所配置的大模型 API，试点前必须在隐私说明中明确告知用户。

## 架构

```text
浏览器
  ├─ 档案 / 会话 / 计划 / 日志 / 复盘 / 反馈
  └─ SSE 流式回答
        │
FastAPI
  ├─ 身份认证、限流、隐私与安全前置检查
  ├─ Hybrid Retriever
  │    ├─ Embedding 召回
  │    ├─ BM25 召回
  │    ├─ RRF 融合
  │    ├─ BGE Reranker
  │    └─ 阈值过滤 / 拒答
  ├─ DeepSeek Responses / Chat Completions
  └─ SQLite
       ├─ users.db：账号、档案、会话、计划、训练日志
       └─ feedback.db：回答快照、反馈、审核状态
```

## 本地运行

建议使用 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

修改 `.env`，至少填写：

```dotenv
DEEPSEEK_API_KEY=...
JWT_SECRET=...
EMBED_MODEL_NAME=./models/fit-sport-emb
RERANK_MODEL_NAME=./models/bge-reranker-v2-m3
DATA_DIR=./data
CHROMA_PATH=./data/chroma_db
```

模型文件较大，不提交到 Git。需要准备以下目录：

```text
models/fit-sport-emb/
models/bge-reranker-v2-m3/
```

启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.backend:app --host 127.0.0.1 --port 8000
```

访问 <http://127.0.0.1:8000>。健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

容器启动：

```powershell
docker compose up --build -d
```

`compose.yaml` 会挂载 `./models` 和 `./data`。容器内的模型路径会自动切换为
`/app/models/...`，因此不要把 Windows 绝对路径写进容器环境。

## 重建知识库

```powershell
python scripts/rebuild_kb.py
```

重建后应在同一套评测集上重新运行检索评测。知识库变更不能让旧指标直接失效。
重建会生成带来源和章节的新文档 ID；如果评测集仍引用旧 `doc_*` ID，必须先人工
重新核对 `gold_doc_ids`，不能直接拿新库结果与旧标签比较。

对于仍使用旧版 `doc_*` ID 的评测集，可以先查看映射报告，再显式写入：

```powershell
python scripts/relabel_eval_ids.py
python scripts/relabel_eval_ids.py --write
```

写入前会自动保留 `evaluation/retrieval.jsonl.before-relabel`。映射使用旧版分块文本与当前
片段的内容包含度，未达到阈值时会停止，不会盲改标签。

查看当前 Chroma collection 和元数据：

```powershell
python scripts/inspect_kb.py
```

## 来源与模型追溯

`docs/knowledge_registry.json` 记录每个知识文件的来源、审核状态、版本和 SHA-256；
`docs/model_manifest.json` 记录部署模型、基座模型、权重哈希和数据版本。当前知识稿
尚未全部标记为 `approved`，模型清单标记为 `needs_retrain_and_review`，不能宣称已
通过专业审核。

项目作者具备专业教练资质时，可以逐段人工检查后记录自审结果：

```powershell
python scripts/review_source.py --source-key training `
  --reviewer "你的姓名" --status approved --confirm
python scripts/review_source.py --source-key safety `
  --reviewer "你的姓名" --status approved --confirm
python scripts/rebuild_kb.py
```

审核脚本会重新计算当前文件 SHA-256，并写入审核人和 UTC 时间。只有实际检查内容后
再执行，不能仅为了绕过生产门禁而直接修改状态。

```powershell
python scripts/audit_provenance.py
python scripts/audit_provenance.py --hash-models
python scripts/audit_provenance.py --production
```

`APP_ENV=production` 时，服务会在启动阶段强制检查来源审核和模型清单；未完成审核或
权重不可追溯时会拒绝启动，避免“内部整理稿”被误宣传为权威医学内容。

## 评测

```powershell
python scripts/evaluate.py
```

脚本直接调用生产使用的 `app.embed_rerank.retrieve()`，比较：

- `vector`
- `hybrid_no_rerank`
- `hybrid`

输出 Recall@K、MRR@K、nDCG@K、Precision@K、bootstrap 95% 置信区间、P95
延迟、分类结果、训练/评测重复检查，以及允许问题误拒、危险问题漏检等安全指标。
默认会排除训练集与评测集之间精确或高度相似的 query，避免指标偏乐观；调试时可加
`--include-leaky` 显式包含。报告写入 `reports/`。

当前可复现结果与解释边界见 `docs/BENCHMARK.md`。

当前评测集规模只适合作为工程回归和小样本验证，不应包装成大规模用户证明。
试用后应把匿名化真实问题按类别加入评测集，并人工复核 gold 文档。

对运行中的服务做端到端回答检查：

```powershell
python scripts/evaluate_generation.py --token "<test-account-jwt>"
```

该脚本会真实调用大模型，并使用 LLM judge 对忠实性、安全、个性化、可执行性和
引用质量打分。报告仍必须人工抽检，不能把模型自评直接当成用户认可。
生成评测默认要求至少 30 条，输出数据指纹、judge 模型和端到端延迟；当前结果只能
作为上线前筛查，仍需独立人工复核。

## 反馈审核

反馈不会自动进入线上提示。管理员可以配置 `ADMIN_TOKEN` 后查看和审核：

```powershell
$headers = @{ "X-Admin-Token" = $env:ADMIN_TOKEN }
Invoke-RestMethod http://127.0.0.1:8000/api/admin/feedback/stats -Headers $headers
```

批准一条反思或点赞示例：

```powershell
$body = @{ answer_id = "..." ; status = "approved" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/admin/feedback/review `
  -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

只有 `FEEDBACK_AUTO_INJECT=1` 且审核状态为 `approved` 的样本才会注入提示。

## 试点指标

试点阶段建议按周记录：

- 首次获得可执行计划率。
- 计划开始率、训练记录完成率和周复盘提交率。
- 7 日回访率。
- 点赞率、点踩原因分布和人工审核确认的错误率。
- 检索拒答率、危险请求拦截率和红旗症状漏检数。
- 首字延迟、完整响应 P95 延迟和单次模型成本。

样本较小时只报告描述性数据，不做显著性结论。

生成不含用户原文的试点汇总：

```powershell
python scripts/pilot_report.py
```

## 数据备份

```powershell
python scripts/backup_data.py
```

脚本使用 SQLite 在线备份 API，在 `backups/` 下生成一致性快照。正式试点应把备份
同步到应用机器之外，并限制备份文件访问权限。

新日志只记录查询哈希和长度，不记录原文。升级前生成的旧 `rag.log` 可能包含历史
问题文本，试点公开发布前应由数据所有者确认并删除或归档旧日志。

## 测试与静态检查

```powershell
pip install -r requirements-dev.txt
pytest -q
ruff check .
node tests/frontend_syntax.mjs
node tests/sql_schema.mjs
node tests/python_balance.mjs
```

其中 `sql_schema.mjs` 需要 Node.js 22+；它只读源码中的建表语句并在内存 SQLite
中验证，不会修改正式数据库。

## 已知限制

- 当前限流和登录失败锁是单进程内存实现；多实例部署应迁移到 Redis。
- SQLite 适合单实例小规模试点；高并发应迁移到 PostgreSQL。
- Reranker 在 CPU 上延迟较高，需要根据真实 P95 选择量化、GPU 或更小模型。
- CPU 环境默认只重排 RRF 前 `RERANK_CANDIDATES_CPU` 个候选，以降低延迟；GPU
  环境使用 `RERANK_CANDIDATES`。上线前应根据固定评测集复核召回率和实际 P95。
- 知识文本仍需要专业审核、来源登记和版本治理。
- 产品指标需要在获得用户授权后通过真实试点收集。

