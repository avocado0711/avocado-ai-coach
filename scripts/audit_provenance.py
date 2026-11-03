"""Validate knowledge-source review status and model-artifact fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from app.config import EMBED_MODEL_NAME, RERANK_MODEL_NAME
from app.provenance import (
    load_model_manifest,
    sha256_file,
    validate_sources,
)


def _artifact_path(reference: str) -> Path | None:
    path = Path(reference)
    if path.is_dir():
        path = path / "model.safetensors"
    return path if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production",
        action="store_true",
        help="Fail unless every knowledge source has documented professional review.",
    )
    parser.add_argument(
        "--hash-models",
        action="store_true",
        help="Hash local model weights and compare them with model_manifest.json.",
    )
    args = parser.parse_args()

    sources = validate_sources(production=args.production)
    models = load_model_manifest()
    hash_errors = []
    if args.hash_models:
        for name, reference in (
            ("embedding", EMBED_MODEL_NAME),
            ("reranker", RERANK_MODEL_NAME),
        ):
            artifact = _artifact_path(reference)
            expected = models["artifacts"].get(name, {}).get("artifact_sha256")
            if artifact is None:
                hash_errors.append(f"{name} 权重文件不存在: {reference}")
            elif expected and sha256_file(artifact).lower() != expected.lower():
                hash_errors.append(f"{name} 权重哈希与模型清单不一致")

    result = {
        "knowledge": sources,
        "models": models,
        "model_hash_errors": hash_errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if sources["errors"] or models["errors"] or hash_errors:
        raise SystemExit(1)
    if args.production and not sources["production_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
