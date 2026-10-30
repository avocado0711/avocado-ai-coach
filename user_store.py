# user_store.py
"""
多用户持久化：用户账号、身体档案、对话历史。
全部用 SQLite（标准库），无需额外部署。
"""
import os
import time
import json
import hashlib
import hmac
import sqlite3
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 改为了移出后的绝对路径
DB_PATH = r"D:\26572\AItools\db\users.db"

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
try:
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
except Exception:
    pass


def _init_db():
    with _lock:
        _conn.executescript("""
        -- 用户账号表
        CREATE TABLE IF NOT EXISTS users (
            user_id      TEXT PRIMARY KEY,
            username     TEXT UNIQUE NOT NULL,
            pwd_hash     TEXT NOT NULL,
            pwd_salt     TEXT NOT NULL,
            created_at   REAL NOT NULL
        );

        -- 用户身体档案（一人一条，可更新）
        CREATE TABLE IF NOT EXISTS user_profile (
            user_id      TEXT PRIMARY KEY,
            height       REAL DEFAULT 0,
            weight       REAL DEFAULT 0,
            age          INTEGER DEFAULT 20,
            gender       TEXT DEFAULT '男',
            train_year   INTEGER DEFAULT 0,
            injury       TEXT DEFAULT '无',
            goal         TEXT DEFAULT '增肌',
            updated_at   REAL NOT NULL
        );

        -- 对话历史（跨会话持久化）
        CREATE TABLE IF NOT EXISTS chat_message (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      TEXT NOT NULL,
            role         TEXT NOT NULL,           -- user / assistant
            content      TEXT NOT NULL,
            answer_id    TEXT,                    -- 仅 assistant 有，用于点赞绑定
            created_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cm_user ON chat_message(user_id, id);
        """)


_init_db()


# ==================== 密码哈希 ====================
def _hash_password(password: str, salt: str = None):
    if salt is None:
        salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120_000)
    return h.hex(), salt


def _verify_password(password: str, pwd_hash: str, salt: str) -> bool:
    h, _ = _hash_password(password, salt)
    return hmac.compare_digest(h, pwd_hash)


# ==================== 用户注册/校验 ====================
def register_user(username: str, password: str) -> dict:
    """注册；返回 {ok, user_id} 或 {ok:False, msg}"""
    username = (username or "").strip()
    if not (2 <= len(username) <= 32):
        return {"ok": False, "msg": "用户名长度需 2-32 字符"}
    if not (6 <= len(password) <= 64):
        return {"ok": False, "msg": "密码长度需 6-64 字符"}

    pwd_hash, salt = _hash_password(password)
    user_id = "u_" + hashlib.md5(f"{username}{time.time()}".encode()).hexdigest()[:16]
    now = time.time()
    with _lock:
        try:
            _conn.execute(
                "INSERT INTO users(user_id,username,pwd_hash,pwd_salt,created_at) VALUES (?,?,?,?,?)",
                (user_id, username, pwd_hash, salt, now),
            )
            _conn.execute(
                "INSERT INTO user_profile(user_id,updated_at) VALUES (?,?)",
                (user_id, now),
            )
            _conn.commit()
            return {"ok": True, "user_id": user_id}
        except sqlite3.IntegrityError:
            return {"ok": False, "msg": "用户名已被占用"}


def verify_user(username: str, password: str) -> dict:
    """登录校验；返回 {ok, user_id} 或 {ok:False, msg}"""
    with _lock:
        row = _conn.execute(
            "SELECT user_id, pwd_hash, pwd_salt FROM users WHERE username=?",
            ((username or "").strip(),),
        ).fetchone()
    if not row:
        return {"ok": False, "msg": "用户名或密码错误"}
    if not _verify_password(password, row["pwd_hash"], row["pwd_salt"]):
        return {"ok": False, "msg": "用户名或密码错误"}
    return {"ok": True, "user_id": row["user_id"]}


def user_exists(user_id: str) -> bool:
    with _lock:
        row = _conn.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone()
    return row is not None


def get_username(user_id: str) -> str:
    with _lock:
        row = _conn.execute("SELECT username FROM users WHERE user_id=?", (user_id,)).fetchone()
    return row["username"] if row else ""


# ==================== 身体档案 ====================
def get_profile(user_id: str) -> dict:
    with _lock:
        row = _conn.execute("SELECT * FROM user_profile WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return {
            "height": 0.0, "weight": 0.0, "age": 20, "gender": "男",
            "train_year": 0, "injury": "无", "goal": "增肌",
            "exists": False,
        }
    d = dict(row)
    # 判断是否"填过"：身高体重都 >0 视为已完善
    d["exists"] = d.get("height", 0) > 0 and d.get("weight", 0) > 0
    return d


def upsert_profile(user_id: str, p: dict) -> dict:
    now = time.time()
    with _lock:
        _conn.execute("""
        INSERT INTO user_profile(user_id,height,weight,age,gender,train_year,injury,goal,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            height=excluded.height, weight=excluded.weight, age=excluded.age,
            gender=excluded.gender, train_year=excluded.train_year,
            injury=excluded.injury, goal=excluded.goal, updated_at=excluded.updated_at
        """, (
            user_id,
            float(p.get("height", 0) or 0),
            float(p.get("weight", 0) or 0),
            int(p.get("age", 20) or 20),
            str(p.get("gender", "男") or "男"),
            int(p.get("train_year", 0) or 0),
            str(p.get("injury", "无") or "无"),
            str(p.get("goal", "增肌") or "增肌"),
            now,
        ))
        _conn.commit()
    return get_profile(user_id)


# ==================== 对话历史 ====================
MAX_HISTORY_ROWS = 100  # 每用户最多保留条数（含 user+assistant）


def add_message(user_id: str, role: str, content: str, answer_id: str = None):
    now = time.time()
    with _lock:
        _conn.execute(
            "INSERT INTO chat_message(user_id,role,content,answer_id,created_at) VALUES (?,?,?,?,?)",
            (user_id, role, content, answer_id, now),
        )
        # 超量裁剪：只保留最近 MAX_HISTORY_ROWS 条
        cnt = _conn.execute(
            "SELECT COUNT(*) AS c FROM chat_message WHERE user_id=?", (user_id,)
        ).fetchone()["c"]
        if cnt > MAX_HISTORY_ROWS:
            cutoff = cnt - MAX_HISTORY_ROWS
            _conn.execute(
                "DELETE FROM chat_message WHERE id IN "
                "(SELECT id FROM chat_message WHERE user_id=? ORDER BY id ASC LIMIT ?)",
                (user_id, cutoff),
            )
        _conn.commit()


def get_messages(user_id: str, limit: int = 30) -> list:
    """取最近 limit 条，按时间正序返回 [{role, content, answer_id, created_at}]"""
    with _lock:
        rows = _conn.execute(
            "SELECT role, content, answer_id, created_at FROM chat_message WHERE user_id=? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "answer_id": r["answer_id"], "created_at": r["created_at"]}
            for r in reversed(rows)]


def clear_messages(user_id: str):
    with _lock:
        _conn.execute("DELETE FROM chat_message WHERE user_id=?", (user_id,))
        _conn.commit()