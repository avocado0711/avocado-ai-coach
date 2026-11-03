"""Create consistent SQLite backups without stopping the application."""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from dotenv import load_dotenv

BASE_DIR = PROJECT_ROOT
load_dotenv(BASE_DIR / ".env")


def configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


DATA_DIR = configured_path("DATA_DIR", BASE_DIR / "data")
USERS_DB_PATH = configured_path("USERS_DB_PATH", DATA_DIR / "users.db")
FEEDBACK_DB_PATH = configured_path("FEEDBACK_DB_PATH", DATA_DIR / "feedback.db")


def backup(source: Path, destination: Path) -> None:
    if not source.exists():
        print(f"跳过不存在的数据库: {source}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_conn, sqlite3.connect(
        destination
    ) as target_conn:
        source_conn.backup(target_conn)
    print(f"已备份: {source} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR
        / "backups"
        / time.strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    backup(USERS_DB_PATH, args.output / "users.db")
    backup(FEEDBACK_DB_PATH, args.output / "feedback.db")


if __name__ == "__main__":
    main()
