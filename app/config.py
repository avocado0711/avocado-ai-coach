"""Application configuration loaded exclusively from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _model_reference(name: str, default: Path) -> str:
    value = os.getenv(name)
    if not value:
        return str(default.resolve())
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith((".", "~")):
        return str(_path_from_env(name, default))
    # Hugging Face model identifiers (for example "BAAI/bge-m3") must remain
    # untouched so the local cache can resolve them.
    return value


APP_ENV = os.getenv("APP_ENV", "development")
POLICY_VERSION = os.getenv("POLICY_VERSION", "2026-11-01")
SOURCE_REGISTRY_PATH = _path_from_env(
    "SOURCE_REGISTRY_PATH",
    BASE_DIR / "docs" / "knowledge_registry.json",
)
MODEL_MANIFEST_PATH = _path_from_env(
    "MODEL_MANIFEST_PATH",
    BASE_DIR / "docs" / "model_manifest.json",
)

# Local application data. The directory is configurable so the project can run
# outside the original author's Windows directories.
DATA_DIR = _path_from_env("DATA_DIR", BASE_DIR / "data")
USERS_DB_PATH = _path_from_env("USERS_DB_PATH", DATA_DIR / "users.db")
FEEDBACK_DB_PATH = _path_from_env("FEEDBACK_DB_PATH", DATA_DIR / "feedback.db")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LLM ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
client = (
    OpenAI(api_key=DEEPSEEK_API_KEY, base_url=LLM_BASE_URL)
    if DEEPSEEK_API_KEY
    else None
)


def require_llm_client():
    if client is None:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，请检查 .env")
    return client

# ---------- Retrieval ----------
EMBED_MODEL_NAME = _model_reference(
    "EMBED_MODEL_NAME",
    BASE_DIR / "models" / "fit-sport-emb",
)
RERANK_MODEL_NAME = _model_reference(
    "RERANK_MODEL_NAME",
    BASE_DIR / "models" / "bge-reranker-v2-m3",
)
CHROMA_PATH = _path_from_env("CHROMA_PATH", DATA_DIR / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sport_knowledge")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "50"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
RERANK_CANDIDATES_CPU = int(os.getenv("RERANK_CANDIDATES_CPU", "6"))
RERANK_BATCH_SIZE = max(1, int(os.getenv("RERANK_BATCH_SIZE", "8")))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
MIN_VECTOR_SCORE = float(os.getenv("MIN_VECTOR_SCORE", "0.35"))
RETRIEVAL_CACHE_SIZE = int(os.getenv("RETRIEVAL_CACHE_SIZE", "256"))

# ---------- Plan adaptation ----------
PLAN_ADAPTATION_LOOKBACK_DAYS = max(
    7,
    int(os.getenv("PLAN_ADAPTATION_LOOKBACK_DAYS", "56")),
)

# ---------- Feedback learning ----------
# Feedback is quarantined by default. It only enters prompts after an admin
# explicitly approves it, preventing accidental reinforcement of bad answers.
FEEDBACK_AUTO_INJECT = os.getenv("FEEDBACK_AUTO_INJECT", "0") == "1"
LEARNING_CACHE_TTL = float(os.getenv("LEARNING_CACHE_TTL", "300"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# ---------- Retention / limits ----------
PENDING_TTL_DAYS = int(os.getenv("PENDING_TTL_DAYS", "7"))
FEEDBACK_RETENTION_DAYS = int(os.getenv("FEEDBACK_RETENTION_DAYS", "365"))
CLEAN_INTERVAL_HOURS = float(os.getenv("CLEAN_INTERVAL_HOURS", "6"))
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
CHAT_LIMIT_PER_MINUTE = int(os.getenv("CHAT_LIMIT_PER_MINUTE", "20"))
CHAT_LIMIT_PER_DAY = int(os.getenv("CHAT_LIMIT_PER_DAY", "200"))
