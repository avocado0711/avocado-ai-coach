"""Rebuild the local Chroma collection with source and section metadata."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    BASE_DIR,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    POLICY_VERSION,
)
from app.provenance import validate_sources

logger = logging.getLogger(__name__)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100


def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 1 <= size:
            current = f"{current}\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        sentences = re.split(r"(?<=[。！？.!?])", paragraph)
        current = current[-overlap:] if current else ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 > size and current:
                chunks.append(current.strip())
                current = current[-overlap:] + sentence
            else:
                current = f"{current}{sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    sections = []
    heading = "总览"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if body:
                sections.append((heading, "\n".join(body).strip()))
            heading = line.lstrip("#").strip() or "总览"
            body = []
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body).strip()))
    return sections


def main() -> None:
    provenance = validate_sources(production=False)
    if not provenance["valid"]:
        raise SystemExit(
            "知识来源注册表无效:\n- " + "\n- ".join(provenance["errors"])
        )
    for warning in provenance["warnings"]:
        logger.warning(warning)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001 - first build has no collection
        logger.info("未找到可覆盖的旧 collection: %s", exc)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "policy_version": POLICY_VERSION},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for source in provenance["sources"]:
        source_key = str(source["source_key"])
        path = Path(str(source["path"]))
        if not path.is_absolute():
            path = BASE_DIR / path
        title = str(source["title"])
        if not path.exists():
            raise FileNotFoundError(path)
        for section_index, (section, section_text) in enumerate(
            split_sections(path.read_text(encoding="utf-8"))
        ):
            for chunk_index, chunk in enumerate(split_into_chunks(section_text)):
                doc_id = f"{source_key}_{section_index}_{chunk_index}"
                ids.append(doc_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "doc_id": doc_id,
                        "source": path.name,
                        "source_title": title,
                        "section": section,
                        "section_index": section_index,
                        "chunk_idx": chunk_index,
                        "policy_version": POLICY_VERSION,
                        "source_origin": source.get("origin", ""),
                        "source_url": source.get("source_url", ""),
                        "evidence_level": source.get("evidence_level", ""),
                        "review_status": source.get("review_status", "draft"),
                        "reviewer": source.get("reviewer") or "",
                        "reviewed_at": source.get("reviewed_at") or "",
                        "source_version": source.get("version", ""),
                        "license": source.get("license", ""),
                        "source_sha256": source.get("sha256", ""),
                    }
                )

    batch_size = 64
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        embeddings = embed_model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=ids[start : start + batch_size],
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=metadatas[start : start + batch_size],
        )
        print(f"写入 {min(start + batch_size, len(documents))}/{len(documents)}")

    print(f"知识库重建完成：{len(documents)} 个片段，collection={COLLECTION_NAME}")
    print(
        "注意：文档 ID 与来源元数据已变化。若评测集仍引用旧 ID，"
        "必须先人工重新标注 gold_doc_ids，再运行 evaluate.py。"
    )
    if provenance["warnings"]:
        print("来源审核警告：" + "；".join(provenance["warnings"]))


if __name__ == "__main__":
    main()
