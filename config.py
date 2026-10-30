# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ---------- LLM ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY 未配置，请检查 .env")

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# ---------- Embedding / Rerank ----------
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "BAAI/bge-m3",
)
RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    r"D:\26572\dunk\bge-reranker-v2-m3",
)
CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(BASE_DIR / "chroma_db"),
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sport_knowledge")