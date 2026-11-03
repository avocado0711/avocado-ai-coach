"""Validation for knowledge-source and model-artifact provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import (
    APP_ENV,
    BASE_DIR,
    EMBED_MODEL_NAME,
    MODEL_MANIFEST_PATH,
    RERANK_MODEL_NAME,
    SOURCE_REGISTRY_PATH,
)

REQUIRED_SOURCE_FIELDS = {
    "source_key",
    "path",
    "title",
    "origin",
    "evidence_level",
    "review_status",
    "version",
}
APPROVED_REVIEW_STATUSES = {"approved"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_registry(
    path: Path = SOURCE_REGISTRY_PATH,
) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("知识来源注册表必须是 JSON 数组")
    return payload


def validate_sources(
    *,
    production: bool | None = None,
) -> dict:
    production = APP_ENV.lower() == "production" if production is None else production
    errors: list[str] = []
    warnings: list[str] = []
    try:
        sources = load_source_registry()
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "production_ready": False,
            "sources": [],
            "errors": [f"无法读取知识来源注册表: {exc}"],
            "warnings": [],
        }

    seen_keys = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            errors.append(f"第 {index} 项不是对象")
            continue
        missing = REQUIRED_SOURCE_FIELDS - set(source)
        if missing:
            errors.append(f"第 {index} 项缺少字段: {', '.join(sorted(missing))}")
            continue
        key = str(source["source_key"])
        if key in seen_keys:
            errors.append(f"source_key 重复: {key}")
        seen_keys.add(key)

        source_path = Path(str(source["path"]))
        if not source_path.is_absolute():
            source_path = BASE_DIR / source_path
        if not source_path.is_file():
            errors.append(f"来源文件不存在: {source_path}")
            continue
        actual_hash = sha256_file(source_path)
        expected_hash = str(source.get("sha256") or "")
        if not expected_hash:
            errors.append(f"{key} 缺少 sha256")
        elif actual_hash.lower() != expected_hash.lower():
            errors.append(f"{key} 内容哈希与注册表不一致")

        review_status = str(source.get("review_status"))
        if review_status not in {"draft", "in_review", "approved", "rejected"}:
            errors.append(f"{key} 的 review_status 无效: {review_status}")
        if review_status != "approved":
            warnings.append(f"{key} 尚未通过专业审核，当前状态为 {review_status}")
        if review_status == "approved" and (
            not source.get("reviewer") or not source.get("reviewed_at")
        ):
            errors.append(f"{key} 标记 approved 但缺少 reviewer/reviewed_at")

    production_ready = bool(sources) and not errors and not warnings
    if production:
        errors.extend(warnings)
        production_ready = False if errors else production_ready
    return {
        "valid": not errors,
        "production_ready": production_ready,
        "sources": sources,
        "errors": errors,
        "warnings": warnings,
    }


def load_model_manifest() -> dict:
    try:
        with MODEL_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"valid": False, "artifacts": {}, "errors": ["模型清单不可用"]}

    errors = []
    artifacts = {}
    expected_paths = {
        "embedding": EMBED_MODEL_NAME,
        "reranker": RERANK_MODEL_NAME,
    }
    for name, expected_path in expected_paths.items():
        artifact = payload.get(name)
        if not isinstance(artifact, dict):
            errors.append(f"模型清单缺少 {name}")
            continue
        if not artifact.get("artifact_sha256"):
            errors.append(f"{name} 缺少 artifact_sha256")
        if not artifact.get("base_model"):
            errors.append(f"{name} 缺少 base_model")
        artifacts[name] = {
            **artifact,
            "configured_reference": expected_path,
        }
    return {
        "valid": not errors,
        "artifacts": artifacts,
        "errors": errors,
        "status": payload.get("status", "unverified"),
    }


def provenance_status() -> dict:
    sources = validate_sources()
    models = load_model_manifest()
    return {
        "knowledge": {
            "valid": sources["valid"],
            "production_ready": sources["production_ready"],
            "total": len(sources["sources"]),
            "drafts": sum(
                source.get("review_status") != "approved"
                for source in sources["sources"]
            ),
            "errors": sources["errors"],
            "warnings": sources["warnings"],
        },
        "models": models,
    }
