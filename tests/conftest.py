import os
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="avocado-tests-"))
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-that-is-long-enough")
os.environ.setdefault("DATA_DIR", str(TEST_ROOT))
os.environ.setdefault("USERS_DB_PATH", str(TEST_ROOT / "users.db"))
os.environ.setdefault("FEEDBACK_DB_PATH", str(TEST_ROOT / "feedback.db"))
os.environ.setdefault("CHROMA_PATH", str(TEST_ROOT / "chroma"))

