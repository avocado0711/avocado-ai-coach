"""Authentication and administrative authorization helpers."""

import os
import secrets
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import Header, HTTPException
from jose import JWTError, jwt

from .config import ADMIN_TOKEN, JWT_EXPIRE_HOURS

JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 未配置，请检查 .env")
if len(JWT_SECRET) < 32:
    raise RuntimeError("JWT_SECRET 至少需要 32 个字符")

_login_failures = defaultdict(list)
_login_lock = threading.Lock()
MAX_FAILS = 5
LOCK_SECONDS = 300


def _clean_old(ip: str) -> None:
    now = time.time()
    _login_failures[ip] = [
        timestamp
        for timestamp in _login_failures[ip]
        if now - timestamp < LOCK_SECONDS
    ]


def check_login_allowed(ip: str) -> None:
    with _login_lock:
        _clean_old(ip)
        if len(_login_failures[ip]) >= MAX_FAILS:
            retry_after = int(_login_failures[ip][0] + LOCK_SECONDS - time.time())
            raise HTTPException(
                status_code=429,
                detail="登录失败次数过多，请 5 分钟后重试",
                headers={"Retry-After": str(max(1, retry_after))},
            )


def record_login_failure(ip: str) -> None:
    with _login_lock:
        _clean_old(ip)
        _login_failures[ip].append(time.time())


def create_token(user_id: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "user_id": user_id,
        "exp": now + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def get_current_user(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = payload["user_id"]
    except (JWTError, KeyError):
        raise HTTPException(status_code=401, detail="token 无效或已过期")

    from .user_store import user_exists

    if not user_exists(user_id):
        raise HTTPException(status_code=401, detail="用户不存在，请重新登录")
    return user_id


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Protect operations that expose cross-user feedback data."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="管理员接口未启用")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="管理员凭据无效")
