# backend.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["NO_WEB_SEARCH"] = "1"

import json
import time
import uuid
import logging
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from auth import (
    create_token,
    get_current_user,
    check_login_allowed,
    record_login_failure,
)
from tools import tools, problem_parse, query_rewrite
from embed_rerank import get_top10_chunks, embed_model, collection   # ★ 复用，不再自加载
from config import client, LLM_MODEL
import user_store

load_dotenv()

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("rag")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
FEEDBACK_DB = r"D:\26572\AItools\db\feedback.db"
# 注意：DB_PATH / EMB_PATH 不再需要，chroma 与 embed 全部由 embed_rerank 统一管理

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY 未配置，请检查 .env")

PENDING_TTL_DAYS = int(os.getenv("PENDING_TTL_DAYS", "7"))
CLEAN_INTERVAL_HOURS = float(os.getenv("CLEAN_INTERVAL_HOURS", "6"))
IO_POOL_SIZE = int(os.getenv("IO_POOL_SIZE", "16"))
LEARNING_CACHE_TTL = float(os.getenv("LEARNING_CACHE_TTL", "300"))

# ==================== LLM ====================
llm = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
_io_pool = ThreadPoolExecutor(max_workers=IO_POOL_SIZE, thread_name_prefix="io")

# ==================== 反馈数据库 ====================
_fb_lock = threading.Lock()
_fb_conn = sqlite3.connect(FEEDBACK_DB, check_same_thread=False)
_fb_conn.row_factory = sqlite3.Row
try:
    _fb_conn.execute("PRAGMA journal_mode=WAL;")
    _fb_conn.execute("PRAGMA synchronous=NORMAL;")
except Exception:
    pass


def _init_feedback_db():
    with _fb_lock:
        _fb_conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            answer_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            query TEXT NOT NULL, answer TEXT NOT NULL,
            rating INTEGER NOT NULL, retrieved TEXT, reflection TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fb_user ON feedback(user_id);
        CREATE INDEX IF NOT EXISTS idx_fb_rating ON feedback(rating);
        CREATE INDEX IF NOT EXISTS idx_fb_created ON feedback(created_at);

        CREATE TABLE IF NOT EXISTS pending_answer (
            answer_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            query TEXT NOT NULL, answer TEXT NOT NULL,
            retrieved TEXT, created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pa_user ON pending_answer(user_id);
        CREATE INDEX IF NOT EXISTS idx_pa_created ON pending_answer(created_at);

        CREATE TABLE IF NOT EXISTS feedback_status (
            answer_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            rating INTEGER NOT NULL, created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fs_user ON feedback_status(user_id);
        """)


_init_feedback_db()


def save_pending_answer(answer_id, user_id, query, answer, retrieved_docs):
    with _fb_lock:
        _fb_conn.execute(
            "INSERT OR REPLACE INTO pending_answer(answer_id,user_id,query,answer,retrieved,created_at) VALUES (?,?,?,?,?,?)",
            (answer_id, user_id, query, answer, json.dumps(retrieved_docs, ensure_ascii=False), time.time()),
        )
        _fb_conn.commit()


def record_feedback(answer_id, user_id, rating):
    with _fb_lock:
        prev = _fb_conn.execute(
            "SELECT rating FROM feedback_status WHERE answer_id=? AND user_id=?",
            (answer_id, user_id),
        ).fetchone()
        if prev is not None:
            return None, True
        row = _fb_conn.execute(
            "SELECT query, answer, retrieved FROM pending_answer WHERE answer_id=? AND user_id=?",
            (answer_id, user_id),
        ).fetchone()
        if not row:
            return None, False
        now = time.time()
        _fb_conn.execute(
            "INSERT OR REPLACE INTO feedback(answer_id,user_id,query,answer,rating,retrieved,reflection,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (answer_id, user_id, row["query"], row["answer"], rating, row["retrieved"], None, now),
        )
        _fb_conn.execute(
            "INSERT OR REPLACE INTO feedback_status(answer_id,user_id,rating,created_at) VALUES (?,?,?,?)",
            (answer_id, user_id, rating, now),
        )
        _fb_conn.execute("DELETE FROM pending_answer WHERE answer_id=?", (answer_id,))
        _fb_conn.commit()
        return {"query": row["query"], "answer": row["answer"], "retrieved": row["retrieved"]}, False


def get_feedback_status(answer_ids, user_id):
    if not answer_ids:
        return {}
    answer_ids = answer_ids[:50]
    placeholders = ",".join("?" * len(answer_ids))
    with _fb_lock:
        rows = _fb_conn.execute(
            f"SELECT answer_id, rating FROM feedback_status WHERE user_id=? AND answer_id IN ({placeholders})",
            [user_id, *answer_ids],
        ).fetchall()
    return {r["answer_id"]: r["rating"] for r in rows}


def update_reflection(answer_id, reflection):
    with _fb_lock:
        _fb_conn.execute("UPDATE feedback SET reflection=? WHERE answer_id=?", (reflection, answer_id))
        _fb_conn.commit()


# ==================== 定期清理 ====================
def _clean_pending_once():
    try:
        cutoff = time.time() - PENDING_TTL_DAYS * 86400
        with _fb_lock:
            cur = _fb_conn.execute("DELETE FROM pending_answer WHERE created_at < ?", (cutoff,))
            _fb_conn.commit()
            n = cur.rowcount
        if n:
            logger.info(f"[clean] 清理过期 pending_answer {n} 条")
    except Exception as e:
        logger.warning(f"[clean] pending 清理失败: {e}")


def _clean_loop():
    while True:
        _clean_pending_once()
        time.sleep(CLEAN_INTERVAL_HOURS * 3600)


_clean_thread = threading.Thread(target=_clean_loop, name="pending-cleaner", daemon=True)


# ==================== lifespan（替代 on_event） ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _clean_pending_once()
    _clean_thread.start()
    logger.info("[startup] 多用户 + 持久化模式已启动")
    yield
    logger.info("[shutdown] 服务已停止")


app = FastAPI(title="私人健身定制训练助手", lifespan=lifespan)

ALLOW_ORIGINS = os.getenv(
    "ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:5500,http://127.0.0.1:8000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 自学习缓存 ====================
class _LearningCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._bad_q = None; self._bad_emb = None; self._bad_ref = None
        self._good_q = None; self._good_a = None; self._good_emb = None
        self._ts = 0.0

    def invalidate(self):
        with self._lock:
            self._ts = 0.0

    def _refresh(self):
        cutoff = time.time() - LEARNING_CACHE_TTL
        with self._lock:
            if self._ts > cutoff and self._bad_emb is not None:
                return
            with _fb_lock:
                bads = _fb_conn.execute(
                    "SELECT query, reflection FROM feedback WHERE rating=-1 AND reflection IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
                goods = _fb_conn.execute(
                    "SELECT query, answer FROM feedback WHERE rating=1 ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
            try:
                if bads:
                    self._bad_q = [r["query"] for r in bads]
                    self._bad_ref = [r["reflection"] for r in bads]
                    self._bad_emb = embed_model.encode(self._bad_q, convert_to_numpy=True,
                                                       normalize_embeddings=True, show_progress_bar=False)
                else:
                    self._bad_q, self._bad_ref, self._bad_emb = [], [], None
                if goods:
                    self._good_q = [r["query"] for r in goods]
                    self._good_a = [r["answer"] for r in goods]
                    self._good_emb = embed_model.encode(self._good_q, convert_to_numpy=True,
                                                        normalize_embeddings=True, show_progress_bar=False)
                else:
                    self._good_q, self._good_a, self._good_emb = [], [], None
                self._ts = time.time()
            except Exception as e:
                logger.warning(f"[learning-cache] refresh failed: {e}")

    def query(self, q_emb, top_k_bad=2, top_k_good=1, th_bad=0.55, th_good=0.7):
        self._refresh()
        with self._lock:
            bads, goods = [], []
            if self._bad_emb is not None and len(self._bad_emb) > 0:
                sims = self._bad_emb @ q_emb
                idx = sims.argsort()[-top_k_bad:][::-1]
                bads = [self._bad_ref[i] for i in idx if sims[i] > th_bad]
            if self._good_emb is not None and len(self._good_emb) > 0:
                sims = self._good_emb @ q_emb
                idx = sims.argsort()[-top_k_good:][::-1]
                goods = [(self._good_q[i], self._good_a[i]) for i in idx if sims[i] > th_good]
            return bads, goods


_learn_cache = _LearningCache()


def build_learning_hint(q_emb):
    if q_emb is None:
        return ""
    try:
        bads, goods = _learn_cache.query(q_emb)
    except Exception as e:
        logger.warning(f"build_learning_hint failed: {e}")
        return ""
    parts = []
    if bads:
        parts.append("【历史踩坑提醒（避免重犯）】\n" + "\n".join(f"- {b}" for b in bads))
    if goods:
        parts.append("【用户认可的回答风格示例】")
        for q, a in goods:
            parts.append(f"问：{q}\n参考答：{a[:300]}")
    return "\n\n".join(parts)


# ==================== 反思生成 ====================
_REFLECTION_SYS = (
    "你是私人健身教练AI团队的质量审核员。\n"
    "用户对下面这条健身回答点了“踩”。请阅读【用户问题】与【被踩回答】，完成两件事：\n"
    "1) 用一句话精准指出这条回答的问题所在（40字以内），必须具体。\n"
    "2) 用一句话提炼一条可执行改进要点（50字以内），必须以“下次应当…”开头。\n"
    "严格按格式输出：\n问题：<第一句>\n改进：<第二句>"
)

_REFLECTION_FEWSHOT = [
    {"role": "user", "content": "用户问题：膝盖有伤能做深蹲吗？\n被踩回答：可以做杠铃深蹲，注意姿势就好。"},
    {"role": "assistant", "content": "问题：未考虑用户膝盖伤病，直接推荐高风险动作。\n改进：下次应当先明确禁忌，将杠铃深蹲替换为无痛区间内的箱式蹲或腿举等低风险替代。"},
]


def _validate_reflection(text):
    if not text or "问题：" not in text or "改进：" not in text:
        return False
    after = text.split("改进：", 1)[1].strip()
    return after.startswith("下次应当") and 10 <= len(after) <= 120


def generate_reflection(query, bad_answer):
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _REFLECTION_SYS},
                    *_REFLECTION_FEWSHOT,
                    {"role": "user", "content": f"用户问题：{query}\n被踩回答：{bad_answer}"},
                ],
                temperature=0.2, timeout=30,
            )
            text = (resp.choices[0].message.content or "").strip()
            if _validate_reflection(text):
                return text
        except Exception as e:
            logger.warning(f"[reflection] failed: {e}")
    return "问题：回答未能充分契合用户问题与档案。\n改进：下次应当结合用户身体数据给出具体动作、组次与休息的表格化方案。"


def _async_reflect(answer_id, query, bad_answer):
    try:
        refl = generate_reflection(query, bad_answer)
        if refl:
            update_reflection(answer_id, refl)
            _learn_cache.invalidate()
    except Exception as e:
        logger.warning(f"[feedback] async reflection failed: {e}")


# ==================== 对话历史（持久化 + 内存缓存） ====================
MAX_HISTORY_LEN = 15
_chat_lock = threading.RLock()
chat_history_pool = OrderedDict()


def _load_history_from_db(user_id: str) -> list:
    rows = user_store.get_messages(user_id, limit=MAX_HISTORY_LEN)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_history(user_id: str) -> list:
    with _chat_lock:
        if user_id in chat_history_pool:
            chat_history_pool.move_to_end(user_id)
            return list(chat_history_pool[user_id])
    hist = _load_history_from_db(user_id)
    with _chat_lock:
        chat_history_pool[user_id] = hist
    return list(hist)


def append_history(user_id: str, role: str, content: str, answer_id: str = None):
    with _chat_lock:
        if user_id not in chat_history_pool:
            chat_history_pool[user_id] = _load_history_from_db(user_id)
        pool = chat_history_pool[user_id]
        pool.append({"role": role, "content": content})
        if len(pool) > MAX_HISTORY_LEN:
            chat_history_pool[user_id] = pool[-MAX_HISTORY_LEN:]
    user_store.add_message(user_id, role, content, answer_id=answer_id)


def build_chat_context(user_id: str, new_msg: str) -> str:
    append_history(user_id, "user", new_msg)
    hist = get_history(user_id)
    return "\n".join(
        f"{'用户' if m['role']=='user' else 'AI教练'}：{m['content']}" for m in hist
    )


def build_messages(user_id: str, system_prompt: str, current_user_content: str):
    messages = [{"role": "system", "content": system_prompt}]
    hist = get_history(user_id)
    for m in hist[:-1] if hist and hist[-1]["role"] == "user" else hist:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": current_user_content})
    return messages


# ==================== 请求结构体 ====================
class UserProfile(BaseModel):
    height: float = Field(default=0.0, ge=0, le=300)
    weight: float = Field(default=0.0, ge=0, le=500)
    age: int = Field(default=20, ge=1, le=120)
    gender: str = "男"
    train_year: int = Field(default=0, ge=0, le=80)
    injury: str = "无"
    goal: str = "增肌"


class ChatReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    profile: UserProfile = None


class AgentChatReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    profile: UserProfile = None


class RegisterReq(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)


class LoginReq(BaseModel):
    username: str = Field(..., min_length=1, max_length=32)
    password: str = Field(..., min_length=1, max_length=64)


class FeedbackReq(BaseModel):
    answer_id: str = Field(..., min_length=1, max_length=64)
    rating: int = Field(..., description="1=赞, -1=踩")


class FeedbackStatusReq(BaseModel):
    answer_ids: list[str] = Field(default_factory=list, max_length=50)


# ==================== 注册 / 登录 ====================
@app.post("/api/register")
async def register(req: RegisterReq, request: Request):
    ip = request.client.host if request.client else "unknown"
    check_login_allowed(ip)
    res = user_store.register_user(req.username, req.password)
    if not res["ok"]:
        raise HTTPException(status_code=400, detail=res["msg"])
    token = create_token(user_id=res["user_id"])
    logger.info(f"[register] user_id={res['user_id']} username={req.username}")
    return {
        "code": 200, "msg": "注册成功",
        "token": token, "user_id": res["user_id"],
        "username": req.username,
        "profile_completed": False,
    }


@app.post("/api/login")
async def login(req: LoginReq, request: Request):
    ip = request.client.host if request.client else "unknown"
    check_login_allowed(ip)
    res = user_store.verify_user(req.username, req.password)
    if not res["ok"]:
        record_login_failure(ip)
        raise HTTPException(status_code=401, detail=res["msg"])
    uid = res["user_id"]
    token = create_token(user_id=uid)
    prof = user_store.get_profile(uid)
    logger.info(f"[login] user_id={uid} ip={ip}")
    return {
        "code": 200, "msg": "success", "token": token,
        "user_id": uid, "username": req.username,
        "profile_completed": prof["exists"],
    }


# ==================== 档案接口 ====================
@app.get("/api/profile")
async def get_profile_api(user_id: str = Depends(get_current_user)):
    prof = user_store.get_profile(user_id)
    return {
        "username": user_store.get_username(user_id),
        "profile": {
            "height": prof["height"], "weight": prof["weight"],
            "age": prof["age"], "gender": prof["gender"],
            "train_year": prof["train_year"], "injury": prof["injury"],
            "goal": prof["goal"],
        },
        "profile_completed": prof["exists"],
    }


@app.post("/api/profile")
async def upsert_profile_api(profile: UserProfile, user_id: str = Depends(get_current_user)):
    updated = await run_in_threadpool(
        user_store.upsert_profile, user_id, profile.model_dump()
    )
    return {"code": 200, "msg": "档案已保存", "profile_completed": updated["exists"]}


# ==================== 历史接口 ====================
@app.get("/api/history")
async def get_history_api(user_id: str = Depends(get_current_user)):
    msgs = await run_in_threadpool(user_store.get_messages, user_id, 50)
    return {"messages": msgs}


@app.post("/api/clear_history")
async def clear_history(user_id: str = Depends(get_current_user)):
    with _chat_lock:
        if user_id in chat_history_pool:
            chat_history_pool[user_id] = []
    await run_in_threadpool(user_store.clear_messages, user_id)
    return {"code": 200, "msg": "对话历史已清空"}


# ==================== 档案文本 ====================
def _profile_text(profile: UserProfile = None, from_db: bool = True, user_id: str = None) -> str:
    p = profile.model_dump() if profile else None
    if p is None and from_db and user_id:
        row = user_store.get_profile(user_id)
        if row.get("exists"):
            p = row
    if not p:
        return "暂无用户身体数据，使用通用健身知识回答。"
    return (
        f"【用户专属身体档案】\n"
        f"性别：{p.get('gender','男')}\n年龄：{p.get('age',20)}岁\n"
        f"身高：{p.get('height',0)}cm\n体重：{p.get('weight',0)}kg\n"
        f"训练年限：{p.get('train_year',0)}年\n"
        f"伤病情况：{p.get('injury','无')}\n训练目标：{p.get('goal','增肌')}\n"
    )


# ==================== 检索 ====================
def _retrieve_sync(query: str, top_k: int, score_threshold: float):
    t0 = time.time()
    results = []
    try:
        chunks = get_top10_chunks(query)
        for c in chunks[:top_k]:
            if isinstance(c, (tuple, list)) and len(c) == 2:
                results.append((str(c[0]), float(c[1])))
            else:
                results.append((str(c), 1.0))
    except Exception as e:
        logger.warning(f"get_top10_chunks failed: {e}, fallback to pure vector search")

    if not results:
        q_vec = embed_model.encode(query, show_progress_bar=False)
        res = collection.query(query_embeddings=[q_vec], n_results=top_k)
        docs = res["documents"][0] if res.get("documents") and res["documents"][0] else []
        results = [(d, 1.0) for d in docs]

    if score_threshold > 0:
        results = [(t, s) for t, s in results if s >= score_threshold]

    cost_ms = (time.time() - t0) * 1000
    logger.info(json.dumps({
        "event": "retrieve", "query": query, "n_hits": len(results),
        "top_scores": [round(s, 4) for _, s in results[:5]],
        "cost_ms": round(cost_ms, 1),
    }, ensure_ascii=False))
    return results


async def retrieve_with_rerank_async(query: str, top_k: int = 5, score_threshold: float = 0.0):
    return await run_in_threadpool(_retrieve_sync, query, top_k, score_threshold)


# ==================== 核心聊天（流式） ====================
SYSTEM_PROMPT = (
    "你是专属私人AI健身教练，仅使用本地健身知识库作答。"
    "拥有完整对话记忆，必须承接上文、支持追问、修改计划、细化计划。"
    "严格根据用户身体数据、训练年限、伤病、目标千人千面定制训练方案。"
    "删去生成文本中无用的符号。禁止通用模板、禁止套话。"
    "如果本地知识库未检索到相关内容，请明确说明，并基于通用健身知识谨慎回答。"
)


@app.post("/api/chat")
async def chat(req: ChatReq, user_id: str = Depends(get_current_user)):
    user_q = req.query.strip()
    profile = req.profile

    build_chat_context(user_id, user_q)

    user_body_info = _profile_text(profile, from_db=True, user_id=user_id)

    retrieved = await retrieve_with_rerank_async(user_q, top_k=5, score_threshold=0.0)
    docs = [t for t, _ in retrieved]
    ref_text = "\n\n".join(docs) if docs else "（本地知识库未检索到相关内容，请基于通用健身知识谨慎回答，并说明未命中知识库）"

    q_emb = await run_in_threadpool(
        lambda: embed_model.encode(user_q, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
    )
    learning_hint = await run_in_threadpool(build_learning_hint, q_emb)
    final_system_prompt = SYSTEM_PROMPT + ("\n\n" + learning_hint if learning_hint else "")

    current_user_content = (
        f"{user_body_info}\n"
        f"【本地知识库资料】\n{ref_text}\n"
        f"【用户问题】\n{user_q}"
    )
    messages = build_messages(user_id, final_system_prompt, current_user_content)
    answer_id = uuid.uuid4().hex

    async def stream_gen():
        answer_buffer = ""
        yield f"event: meta\ndata: {json.dumps({'answer_id': answer_id}, ensure_ascii=False)}\n\n"
        try:
            resp = await run_in_threadpool(
                llm.chat.completions.create,
                model="deepseek-chat",
                messages=messages,
                stream=True,
                timeout=60,
            )
            it = iter(resp)
            while True:
                chunk = await run_in_threadpool(next, it, None)
                if chunk is None:
                    break
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    answer_buffer += content
                    yield f"data: {content}\n\n"
        except Exception as e:
            logger.exception(f"stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            if answer_buffer:
                append_history(user_id, "assistant", answer_buffer, answer_id=answer_id)
                await run_in_threadpool(
                    save_pending_answer, answer_id, user_id, user_q, answer_buffer, docs
                )
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


# ==================== Agent 调度 ====================
def _agent_dispatch_sync(user_q: str, history_text: str):
    messages = [
        {"role": "system", "content": "你只能使用本地知识库回答，禁止任何网页搜索。仅做问题解析和改写，绝不联网。"},
        {"role": "user", "content": f"用户问题：{user_q}\n对话历史：{history_text}"},
    ]
    max_round = 5
    final_query = user_q
    deadline = time.time() + 90

    for _ in range(max_round):
        if time.time() > deadline:
            return {"type": "query", "content": user_q}
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=messages, tools=tools,
                tool_choice="auto", timeout=60,
            )
        except Exception as e:
            logger.warning(f"agent_dispatch LLM failed: {e}, fallback")
            return {"type": "query", "content": user_q}

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"type": "direct", "content": msg.content}

        for call in msg.tool_calls:
            fname = call.function.name
            try:
                args = json.loads(call.function.arguments) if call.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            if fname == "problem_parse":
                ret = problem_parse(**args)
            elif fname == "query_rewrite":
                ret = query_rewrite(**args)
                final_query = ret.get("rewritten_query", user_q)
            else:
                ret = {"result": "仅使用本地知识库回答，无需联网"}
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(ret, ensure_ascii=False)})
    return {"type": "query", "content": final_query}


AGENT_SYSTEM_PROMPT = (
    "承接历史对话，支持追问细化。"
    "只使用本地专属健身知识库作答，禁止联网、禁止编造。"
    "给出精准、个性化、可落地的训练方案。输出的训练方案用表格方式呈现。"
)


@app.post("/api/agent_chat")
async def agent_chat(req: AgentChatReq, user_id: str = Depends(get_current_user)):
    user_q = req.query.strip()
    full_context = build_chat_context(user_id, user_q)

    dispatch_res = await run_in_threadpool(_agent_dispatch_sync, user_q, full_context)
    clean_query = (dispatch_res["content"] or "").strip()

    if dispatch_res["type"] == "direct":
        answer = clean_query
        answer_id = uuid.uuid4().hex
        append_history(user_id, "assistant", answer, answer_id=answer_id)
        await run_in_threadpool(save_pending_answer, answer_id, user_id, user_q, answer, [])
        return {"answer": answer, "answer_id": answer_id}

    retrieved = await retrieve_with_rerank_async(clean_query, top_k=10, score_threshold=0.0)
    docs = [t for t, _ in retrieved]
    context_text = "\n\n".join(docs) if docs else "（本地知识库未检索到相关内容）"

    user_info = _profile_text(req.profile, from_db=True, user_id=user_id)

    q_emb = await run_in_threadpool(
        lambda: embed_model.encode(clean_query, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
    )
    learning_hint = await run_in_threadpool(build_learning_hint, q_emb)
    final_system_prompt = AGENT_SYSTEM_PROMPT + ("\n\n" + learning_hint if learning_hint else "")

    current_user_content = (
        f"{user_info}\n"
        f"【参考资料】\n{context_text}\n"
        f"【用户问题】\n{clean_query}"
    )
    messages = build_messages(user_id, final_system_prompt, current_user_content)

    try:
        final_resp = await run_in_threadpool(
            client.chat.completions.create,
            model=LLM_MODEL, messages=messages, timeout=60,
        )
        answer = final_resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception(f"agent_chat final LLM failed: {e}")
        raise HTTPException(status_code=500, detail=f"模型调用失败: {str(e)}")

    answer_id = uuid.uuid4().hex
    append_history(user_id, "assistant", answer, answer_id=answer_id)
    await run_in_threadpool(save_pending_answer, answer_id, user_id, user_q, answer, docs)
    return {"answer": answer, "answer_id": answer_id}


# ==================== 反馈接口 ====================
@app.post("/api/feedback")
async def feedback(req: FeedbackReq, user_id: str = Depends(get_current_user)):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating 只能为 1 或 -1")

    row, already = record_feedback(req.answer_id, user_id, req.rating)

    if already:
        return {"code": 200, "msg": "already_rated", "already_rated": True}
    if not row:
        return {"code": 200, "msg": "记录不存在或已过期", "already_rated": False}

    if req.rating == -1:
        threading.Thread(
            target=_async_reflect,
            args=(req.answer_id, row["query"], row["answer"]),
            daemon=True,
        ).start()
        logger.info(f"[feedback] 已记录点踩，后台生成反思 answer_id={req.answer_id}")

    return {"code": 200, "msg": "反馈已记录", "already_rated": False}


@app.post("/api/feedback/status")
async def feedback_status(req: FeedbackStatusReq, user_id: str = Depends(get_current_user)):
    status = get_feedback_status(req.answer_ids, user_id)
    return {"status": status}


@app.get("/api/feedback/stats")
async def feedback_stats(user_id: str = Depends(get_current_user)):
    with _fb_lock:
        total = _fb_conn.execute("SELECT COUNT(*) AS c FROM feedback").fetchone()["c"]
        likes = _fb_conn.execute("SELECT COUNT(*) AS c FROM feedback WHERE rating=1").fetchone()["c"]
        dislikes = _fb_conn.execute("SELECT COUNT(*) AS c FROM feedback WHERE rating=-1").fetchone()["c"]
        pending = _fb_conn.execute("SELECT COUNT(*) AS c FROM pending_answer").fetchone()["c"]
        recent = _fb_conn.execute(
            "SELECT rating, query, reflection FROM feedback ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
    return {
        "total": total, "likes": likes, "dislikes": dislikes,
        "pending": pending, "recent": [dict(r) for r in recent],
    }


 # ==================== 静态页面 ====================
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def home():
        return FileResponse(os.path.join(STATIC_DIR, "staticindex.html"))
else:
    logger.warning(f"static dir not found: {STATIC_DIR}, 跳过静态挂载")
# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)