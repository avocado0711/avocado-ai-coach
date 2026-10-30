# auth.py（在原有基础上只改 get_current_user 的返回处）
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

from dotenv import load_dotenv
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, Header

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD")
ALGORITHM = "HS256"
EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 未配置，请检查 .env")
if not ACCESS_PASSWORD:
    raise RuntimeError("ACCESS_PASSWORD 未配置，请检查 .env")

_login_failures = defaultdict(list)
MAX_FAILS = 5
LOCK_SECONDS = 30


def _clean_old(ip: str):
    now = time.time()
    _login_failures[ip] = [t for t in _login_failures[ip] if now - t < LOCK_SECONDS]


def check_login_allowed(ip: str):
    _clean_old(ip)
    if len(_login_failures[ip]) >= MAX_FAILS:
        raise HTTPException(status_code=429, detail="登录失败次数过多，请稍后再试")


def record_login_failure(ip: str):
    _clean_old(ip)
    _login_failures[ip].append(time.time())


def verify_password(input_password: str) -> bool:
    import secrets
    return secrets.compare_digest(str(input_password), str(ACCESS_PASSWORD))


def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def get_current_user(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Bearer token")
    token = authorization.removeprefix("Bearer ")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        uid = payload["user_id"]
    except JWTError:
        raise HTTPException(status_code=401, detail="token 无效或已过期")

    # ★ 新增：校验用户是否真实存在（防止伪造 user_id 越权）
    from user_store import user_exists
    if not user_exists(uid):
        raise HTTPException(status_code=401, detail="用户不存在，请重新登录")
    return uid