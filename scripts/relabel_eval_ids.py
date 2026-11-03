"""Map legacy eval doc ids to section-aware ids after a knowledge rebuild.

The legacy collection used doc_0 ... doc_N from one full-text chunking pass.
This utility recreates those chunks deterministically and maps each referenced
chunk to the most similar chunks in the current collection.
"""

from __future__ import annotations

import argparse
import json
import re

import chromadb
from _bootstrap import PROJECT_ROOT

from app.config import CHROMA_PATH, COLLECTION_NAME

EVAL_FILE = PROJECT_ROOT / "evaluation" / "retrieval.jsonl"
SOURCE_FILE = PROJECT_ROOT / "docs" / "strength_training.txt"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100


def legacy_chunks(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > CHUNK_SIZE:
            sentences = re.split(r"(?<=[。！？!?])", paragraph)
            for sentence in sentences:
                if len(current) + len(sentence) > CHUNK_SIZE:
                    if current:
                        chunks.append(current.strip())
                        current = current[-CHUNK_OVERLAP:] + sentence
                    else:
                        current = sentence
                else:
                    current += sentence
        elif len(current) + len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = current[-CHUNK_OVERLAP:] + paragraph
            else:
                current = paragraph
        else:
            current += "\n" + paragraph
    if current.strip():
        chunks.append(current.strip())
    return chunks


def bigrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    return {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--top", type=int, default=2)
    args = parser.parse_args()

    old_chunks = legacy_chunks(SOURCE_FILE.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    current = collection.get(include=["documents", "metadatas"])
    current_bigrams = [bigrams(document or "") for document in current["documents"]]
    eval_items = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    used_ids = {
        doc_id
        for item in eval_items
        for doc_id in item.get("gold_doc_ids", [])
    }
    missing_legacy = sorted(
        doc_id
        for doc_id in used_ids
        if int(doc_id.removeprefix("doc_")) >= len(old_chunks)
    )
    if missing_legacy:
        raise SystemExit(f"无法还原旧片段: {missing_legacy}")

    mapping = {}
    confidence = {}
    for doc_id in sorted(used_ids, key=lambda value: int(value.removeprefix("doc_"))):
        index = int(doc_id.removeprefix("doc_"))
        old = bigrams(old_chunks[index])
        candidates = []
        for current_index, candidate in enumerate(current_bigrams):
            overlap = len(old & candidate) / max(1, len(old))
            if overlap >= args.threshold:
                candidates.append((overlap, current["ids"][current_index]))
        candidates.sort(reverse=True)
        selected = [item[1] for item in candidates[: args.top]]
        mapping[doc_id] = selected
        confidence[doc_id] = [
            {"new_id": new_id, "containment": round(score, 3)}
            for score, new_id in candidates[: args.top]
        ]

    unresolved = [doc_id for doc_id, ids in mapping.items() if not ids]
    print(f"旧片段数: {len(old_chunks)}，当前片段数: {collection.count()}")
    print(f"评测引用旧 ID: {len(mapping)}，未映射: {unresolved}")
    for doc_id in mapping:
        print(doc_id, "->", confidence[doc_id])
    if unresolved:
        raise SystemExit("存在未映射的 gold_doc_id，未修改评测集")

    if args.write:
        backup = EVAL_FILE.with_suffix(".jsonl.before-relabel")
        if not backup.exists():
            backup.write_text(EVAL_FILE.read_text(encoding="utf-8"), encoding="utf-8")
        updated = []
        for item in eval_items:
            old_gold = item.get("gold_doc_ids", [])
            new_gold = []
            for doc_id in old_gold:
                for mapped_id in mapping[doc_id]:
                    if mapped_id not in new_gold:
                        new_gold.append(mapped_id)
            item["gold_doc_ids"] = new_gold
            item["gold_doc_id"] = new_gold[0] if new_gold else None
            updated.append(item)
        EVAL_FILE.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in updated)
            + "\n",
            encoding="utf-8",
        )
        print(f"已更新: {EVAL_FILE}")
        print(f"原文件备份: {backup}")


if __name__ == "__main__":
    main()
