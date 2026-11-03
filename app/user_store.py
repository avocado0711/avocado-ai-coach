"""SQLite persistence for accounts, profiles, plans and workout logs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid

from .config import POLICY_VERSION, USERS_DB_PATH
from .injuries import normalize_injury_text

DB_PATH = USERS_DB_PATH
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
_conn.row_factory = sqlite3.Row
_conn.execute("PRAGMA journal_mode=WAL;")
_conn.execute("PRAGMA synchronous=NORMAL;")
_conn.execute("PRAGMA foreign_keys=ON;")

PASSWORD_ITERATIONS = 310_000
LEGACY_PASSWORD_ITERATIONS = 120_000
MAX_HISTORY_ROWS = 100


def _column_names(table: str) -> set[str]:
    return {row["name"] for row in _conn.execute(f"PRAGMA table_info({table})")}


def _ensure_column(table: str, column: str, definition: str) -> None:
    if column not in _column_names(table):
        _conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _init_db() -> None:
    with _lock:
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id      TEXT PRIMARY KEY,
                username     TEXT UNIQUE NOT NULL,
                pwd_hash     TEXT NOT NULL,
                pwd_salt     TEXT NOT NULL,
                created_at   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_profile (
                user_id        TEXT PRIMARY KEY,
                height         REAL DEFAULT 0,
                weight         REAL DEFAULT 0,
                age            INTEGER DEFAULT 20,
                gender         TEXT DEFAULT '男',
                train_year     INTEGER DEFAULT 0,
                injury         TEXT DEFAULT '无',
                goal           TEXT DEFAULT '增肌',
                training_days  INTEGER DEFAULT 3,
                session_minutes INTEGER DEFAULT 60,
                updated_at     REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_message (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                role         TEXT NOT NULL,
                content      TEXT NOT NULL,
                answer_id    TEXT,
                created_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cm_user ON chat_message(user_id, id);

            CREATE TABLE IF NOT EXISTS user_consent (
                user_id      TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                accepted_at  REAL NOT NULL,
                PRIMARY KEY (user_id, policy_version)
            );

            CREATE TABLE IF NOT EXISTS training_plan (
                plan_id      TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                title        TEXT NOT NULL,
                goal         TEXT NOT NULL,
                weeks        INTEGER NOT NULL,
                status       TEXT NOT NULL DEFAULT 'active',
                plan_json    TEXT NOT NULL,
                parent_plan_id TEXT,
                version      INTEGER NOT NULL DEFAULT 1,
                change_reason TEXT NOT NULL DEFAULT '',
                created_at   REAL NOT NULL,
                updated_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_plan_user
                ON training_plan(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS workout_log (
                log_id       TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                plan_id      TEXT,
                workout_date TEXT NOT NULL,
                exercise     TEXT NOT NULL,
                sets         INTEGER NOT NULL DEFAULT 0,
                reps         INTEGER NOT NULL DEFAULT 0,
                weight_kg    REAL NOT NULL DEFAULT 0,
                rpe          REAL,
                pain         INTEGER NOT NULL DEFAULT 0,
                completed    INTEGER NOT NULL DEFAULT 0,
                notes        TEXT NOT NULL DEFAULT '',
                created_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_workout_user
                ON workout_log(user_id, workout_date DESC);

            CREATE TABLE IF NOT EXISTS weekly_checkin (
                checkin_id   TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                plan_id      TEXT,
                week         INTEGER NOT NULL,
                adherence    INTEGER NOT NULL,
                energy       INTEGER NOT NULL,
                pain         INTEGER NOT NULL DEFAULT 0,
                notes        TEXT NOT NULL DEFAULT '',
                created_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_checkin_user
                ON weekly_checkin(user_id, created_at DESC);
            """
        )
        _ensure_column("user_profile", "training_days", "INTEGER DEFAULT 3")
        _ensure_column("user_profile", "session_minutes", "INTEGER DEFAULT 60")
        _ensure_column("training_plan", "parent_plan_id", "TEXT")
        _ensure_column("training_plan", "version", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(
            "training_plan",
            "change_reason",
            "TEXT NOT NULL DEFAULT ''",
        )
        _conn.commit()


_init_db()


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${digest}", salt


def _verify_password(password: str, stored: str, salt: str) -> bool:
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, expected = stored.split("$", 2)
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt),
                int(iterations),
            ).hex()
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(digest, expected)

    # Backward compatibility for accounts created before the iterative hash
    # metadata was stored alongside the digest.
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        LEGACY_PASSWORD_ITERATIONS,
    ).hex()
    return hmac.compare_digest(digest, stored)


_DUMMY_HASH, _DUMMY_SALT = _hash_password("not-a-real-user-password")


def register_user(
    username: str,
    password: str,
    consent_version: str = POLICY_VERSION,
) -> dict:
    username = (username or "").strip()
    if not (2 <= len(username) <= 32):
        return {"ok": False, "msg": "用户名长度需为 2-32 个字符"}
    if any(character.isspace() or ord(character) < 32 for character in username):
        return {"ok": False, "msg": "用户名不能包含空白或控制字符"}
    if not re.fullmatch(r"[A-Za-z0-9_.\-\u4e00-\u9fff]+", username):
        return {"ok": False, "msg": "用户名只能包含中文、字母、数字、下划线、点和连字符"}
    if not (8 <= len(password) <= 64):
        return {"ok": False, "msg": "密码长度需为 8-64 个字符"}

    pwd_hash, salt = _hash_password(password)
    user_id = "u_" + secrets.token_urlsafe(16)
    now = time.time()
    with _lock:
        try:
            _conn.execute(
                "INSERT INTO users(user_id,username,pwd_hash,pwd_salt,created_at) "
                "VALUES (?,?,?,?,?)",
                (user_id, username, pwd_hash, salt, now),
            )
            _conn.execute(
                "INSERT INTO user_profile(user_id,updated_at) VALUES (?,?)",
                (user_id, now),
            )
            _conn.execute(
                "INSERT INTO user_consent(user_id,policy_version,accepted_at) "
                "VALUES (?,?,?)",
                (user_id, consent_version, now),
            )
            _conn.commit()
            return {"ok": True, "user_id": user_id}
        except sqlite3.IntegrityError:
            _conn.rollback()
            return {"ok": False, "msg": "用户名已被占用"}


def verify_user(username: str, password: str) -> dict:
    with _lock:
        row = _conn.execute(
            "SELECT user_id, pwd_hash, pwd_salt FROM users WHERE username=?",
            ((username or "").strip(),),
        ).fetchone()
    if not row:
        _verify_password(password, _DUMMY_HASH, _DUMMY_SALT)
        return {"ok": False, "msg": "用户名或密码错误"}
    if not _verify_password(password, row["pwd_hash"], row["pwd_salt"]):
        return {"ok": False, "msg": "用户名或密码错误"}
    if not row["pwd_hash"].startswith("pbkdf2_sha256$"):
        upgraded_hash, upgraded_salt = _hash_password(password)
        with _lock:
            _conn.execute(
                "UPDATE users SET pwd_hash=?,pwd_salt=? WHERE user_id=?",
                (upgraded_hash, upgraded_salt, row["user_id"]),
            )
            _conn.commit()
    return {"ok": True, "user_id": row["user_id"]}


def verify_user_by_id(user_id: str, password: str) -> bool:
    with _lock:
        row = _conn.execute(
            "SELECT pwd_hash, pwd_salt FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if not row:
        _verify_password(password, _DUMMY_HASH, _DUMMY_SALT)
        return False
    return _verify_password(password, row["pwd_hash"], row["pwd_salt"])


def user_exists(user_id: str) -> bool:
    with _lock:
        return (
            _conn.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone()
            is not None
        )


def get_username(user_id: str) -> str:
    with _lock:
        row = _conn.execute(
            "SELECT username FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    return row["username"] if row else ""


def get_consent(user_id: str) -> dict | None:
    with _lock:
        row = _conn.execute(
            "SELECT policy_version, accepted_at FROM user_consent "
            "WHERE user_id=? ORDER BY accepted_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def record_consent(
    user_id: str,
    policy_version: str = POLICY_VERSION,
) -> dict:
    accepted_at = time.time()
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO user_consent(user_id,policy_version,accepted_at) "
            "VALUES (?,?,?)",
            (user_id, policy_version, accepted_at),
        )
        _conn.commit()
    return {"policy_version": policy_version, "accepted_at": accepted_at}


def get_profile(user_id: str) -> dict:
    with _lock:
        row = _conn.execute(
            "SELECT * FROM user_profile WHERE user_id=?", (user_id,)
        ).fetchone()
    if not row:
        return {
            "height": 0.0,
            "weight": 0.0,
            "age": 20,
            "gender": "男",
            "train_year": 0,
            "injury": "无",
            "goal": "增肌",
            "training_days": 3,
            "session_minutes": 60,
            "exists": False,
        }
    result = dict(row)
    result["exists"] = result.get("height", 0) > 0 and result.get("weight", 0) > 0
    return result


def upsert_profile(user_id: str, profile: dict) -> dict:
    now = time.time()
    with _lock:
        _conn.execute(
            """
            INSERT INTO user_profile(
                user_id,height,weight,age,gender,train_year,injury,goal,
                training_days,session_minutes,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                height=excluded.height,
                weight=excluded.weight,
                age=excluded.age,
                gender=excluded.gender,
                train_year=excluded.train_year,
                injury=excluded.injury,
                goal=excluded.goal,
                training_days=excluded.training_days,
                session_minutes=excluded.session_minutes,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                float(profile.get("height", 0) or 0),
                float(profile.get("weight", 0) or 0),
                int(profile.get("age", 20) or 20),
                str(profile.get("gender", "男") or "男"),
                int(profile.get("train_year", 0) or 0),
                normalize_injury_text(profile.get("injury", "无")),
                str(profile.get("goal", "增肌") or "增肌"),
                max(1, min(7, int(profile.get("training_days", 3) or 3))),
                max(15, min(180, int(profile.get("session_minutes", 60) or 60))),
                now,
            ),
        )
        _conn.commit()
    return get_profile(user_id)


def add_message(
    user_id: str,
    role: str,
    content: str,
    answer_id: str | None = None,
) -> None:
    now = time.time()
    with _lock:
        _conn.execute(
            "INSERT INTO chat_message(user_id,role,content,answer_id,created_at) "
            "VALUES (?,?,?,?,?)",
            (user_id, role, content, answer_id, now),
        )
        count = _conn.execute(
            "SELECT COUNT(*) AS c FROM chat_message WHERE user_id=?", (user_id,)
        ).fetchone()["c"]
        if count > MAX_HISTORY_ROWS:
            cutoff = count - MAX_HISTORY_ROWS
            _conn.execute(
                "DELETE FROM chat_message WHERE id IN "
                "(SELECT id FROM chat_message WHERE user_id=? ORDER BY id ASC LIMIT ?)",
                (user_id, cutoff),
            )
        _conn.commit()


def get_messages(user_id: str, limit: int = 30) -> list:
    with _lock:
        rows = _conn.execute(
            "SELECT role, content, answer_id, created_at FROM chat_message "
            "WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def clear_messages(user_id: str) -> None:
    with _lock:
        _conn.execute("DELETE FROM chat_message WHERE user_id=?", (user_id,))
        _conn.commit()


def create_plan(
    user_id: str,
    title: str,
    goal: str,
    weeks: int,
    plan: dict,
    *,
    parent_plan_id: str | None = None,
    change_reason: str = "",
) -> dict:
    plan_id = "p_" + uuid.uuid4().hex
    now = time.time()
    with _lock:
        active = _conn.execute(
            "SELECT plan_id,version FROM training_plan WHERE user_id=? "
            "AND status='active' ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        resolved_parent_id = parent_plan_id or (active["plan_id"] if active else None)
        if resolved_parent_id:
            parent = _conn.execute(
                "SELECT version FROM training_plan WHERE user_id=? AND plan_id=?",
                (user_id, resolved_parent_id),
            ).fetchone()
            version = int(parent["version"]) + 1 if parent else 1
        else:
            version = 1
        _conn.execute(
            "UPDATE training_plan SET status='archived', updated_at=? "
            "WHERE user_id=? AND status='active'",
            (now, user_id),
        )
        _conn.execute(
            "INSERT INTO training_plan("
            "plan_id,user_id,title,goal,weeks,status,plan_json,parent_plan_id,"
            "version,change_reason,created_at,updated_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                plan_id,
                user_id,
                title,
                goal,
                weeks,
                "active",
                json.dumps(plan, ensure_ascii=False),
                resolved_parent_id,
                version,
                change_reason[:1000],
                now,
                now,
            ),
        )
        _conn.commit()
    return get_plan(user_id, plan_id)


def get_plan(user_id: str, plan_id: str | None = None) -> dict | None:
    with _lock:
        if plan_id:
            row = _conn.execute(
                "SELECT * FROM training_plan WHERE user_id=? AND plan_id=?",
                (user_id, plan_id),
            ).fetchone()
        else:
            row = _conn.execute(
                "SELECT * FROM training_plan WHERE user_id=? AND status='active' "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["plan"] = json.loads(result.pop("plan_json"))
    return result


def add_workout_log(user_id: str, data: dict) -> dict:
    log_id = "w_" + uuid.uuid4().hex
    created_at = time.time()
    with _lock:
        _conn.execute(
            """
            INSERT INTO workout_log(
                log_id,user_id,plan_id,workout_date,exercise,sets,reps,
                weight_kg,rpe,pain,completed,notes,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                log_id,
                user_id,
                data.get("plan_id"),
                data["workout_date"],
                data["exercise"],
                int(data.get("sets", 0)),
                int(data.get("reps", 0)),
                float(data.get("weight_kg", 0) or 0),
                data.get("rpe"),
                int(data.get("pain", 0)),
                1 if data.get("completed") else 0,
                str(data.get("notes", "")),
                created_at,
            ),
        )
        _conn.commit()
    return {"log_id": log_id, "created_at": created_at, **data}


def get_workouts(user_id: str, limit: int = 100) -> list:
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM workout_log WHERE user_id=? "
            "ORDER BY workout_date DESC, created_at DESC LIMIT ?",
            (user_id, min(limit, 500)),
        ).fetchall()
    return [dict(row) for row in rows]


def get_checkins(user_id: str, limit: int = 100) -> list:
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM weekly_checkin WHERE user_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, min(max(1, limit), 500)),
        ).fetchall()
    return [dict(row) for row in rows]


def add_checkin(user_id: str, data: dict) -> dict:
    checkin_id = "c_" + uuid.uuid4().hex
    created_at = time.time()
    with _lock:
        _conn.execute(
            """
            INSERT INTO weekly_checkin(
                checkin_id,user_id,plan_id,week,adherence,energy,pain,notes,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                checkin_id,
                user_id,
                data.get("plan_id"),
                int(data.get("week", 1)),
                int(data.get("adherence", 0)),
                int(data.get("energy", 3)),
                int(data.get("pain", 0)),
                str(data.get("notes", "")),
                created_at,
            ),
        )
        _conn.commit()
    return {"checkin_id": checkin_id, "created_at": created_at, **data}


def export_user_data(user_id: str) -> dict:
    with _lock:
        user = _conn.execute(
            "SELECT user_id,username,created_at FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        plans = _conn.execute(
            "SELECT * FROM training_plan WHERE user_id=? ORDER BY created_at",
            (user_id,),
        ).fetchall()
        workouts = _conn.execute(
            "SELECT * FROM workout_log WHERE user_id=? ORDER BY created_at",
            (user_id,),
        ).fetchall()
        checkins = _conn.execute(
            "SELECT * FROM weekly_checkin WHERE user_id=? ORDER BY created_at",
            (user_id,),
        ).fetchall()
        consent = _conn.execute(
            "SELECT * FROM user_consent WHERE user_id=?", (user_id,)
        ).fetchall()
    plan_rows = []
    for row in plans:
        item = dict(row)
        item["plan"] = json.loads(item.pop("plan_json"))
        plan_rows.append(item)
    return {
        "account": dict(user) if user else None,
        "profile": get_profile(user_id),
        "messages": get_messages(user_id, limit=MAX_HISTORY_ROWS),
        "plans": plan_rows,
        "workouts": [dict(row) for row in workouts],
        "checkins": [dict(row) for row in checkins],
        "consent": [dict(row) for row in consent],
    }


def delete_user(user_id: str) -> None:
    with _lock:
        for table in (
            "weekly_checkin",
            "workout_log",
            "training_plan",
            "chat_message",
            "user_profile",
            "user_consent",
            "users",
        ):
            _conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        _conn.commit()
