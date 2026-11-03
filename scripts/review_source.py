"""Record an explicit professional review for one knowledge source."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import _bootstrap  # noqa: F401

from app.config import BASE_DIR, SOURCE_REGISTRY_PATH
from app.provenance import load_source_registry, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--status",
        choices=("approved", "rejected"),
        default="approved",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm that the named reviewer inspected the current file contents.",
    )
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("请先人工检查内容，再添加 --confirm 记录审核结果。")

    sources = load_source_registry()
    source = next(
        (item for item in sources if item.get("source_key") == args.source_key),
        None,
    )
    if source is None:
        raise SystemExit(f"未找到来源: {args.source_key}")
    source_path = BASE_DIR / str(source["path"])
    if not source_path.is_file():
        raise SystemExit(f"来源文件不存在: {source_path}")

    source["sha256"] = sha256_file(source_path)
    source["review_status"] = args.status
    source["reviewer"] = args.reviewer.strip()
    source["reviewed_at"] = datetime.now(UTC).isoformat()
    source["review_method"] = "author_professional_self_review"
    source["review_note"] = args.note.strip()
    SOURCE_REGISTRY_PATH.write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.source_key}: {args.status}, reviewer={source['reviewer']}, "
        f"sha256={source['sha256']}"
    )
    print("请重新运行 scripts/rebuild_kb.py，使审核状态进入知识库元数据。")


if __name__ == "__main__":
    main()
