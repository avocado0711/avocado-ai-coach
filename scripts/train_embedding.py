"""Train a fitness-domain triplet embedding model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import warnings
from datetime import UTC, datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from _bootstrap import PROJECT_ROOT
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fitness triplet embedding model")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "train_triplets.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "fit-sport-emb",
    )
    parser.add_argument("--base-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    return parser.parse_args()


def load_data(path: Path) -> list[dict]:
    items = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        missing = {"query", "positive", "negative"} - set(item)
        if missing:
            raise ValueError(f"{path.name}:{line_number} 缺少字段 {sorted(missing)}")
        items.append(item)
    if len(items) < 20:
        raise ValueError("训练三元组太少，至少需要 20 条")
    return items


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    all_data = load_data(args.data)
    random.shuffle(all_data)
    split_index = max(1, int(len(all_data) * args.train_ratio))
    train_data = all_data[:split_index]
    val_data = all_data[split_index:]
    if not val_data:
        raise ValueError("验证集为空，请降低 train-ratio")

    train_examples = [
        InputExample(
            texts=[item["query"], item["positive"], item["negative"]]
        )
        for item in train_data
    ]
    evaluator = TripletEvaluator(
        [item["query"] for item in val_data],
        [item["positive"] for item in val_data],
        [item["negative"] for item in val_data],
    )

    model = SentenceTransformer(args.base_model)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )
    train_loss = losses.TripletLoss(model)
    print("训练前验证集指标：")
    evaluator(model)
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)

    args.output.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(args.output),
        save_best_model=True,
    )
    model_file = args.output / "model.safetensors"
    manifest = {
        "base_model": args.base_model,
        "data": str(args.data),
        "dataset_sha256": sha256_file(args.data),
        "dataset_rows": len(all_data),
        "train_size": len(train_data),
        "validation_size": len(val_data),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "created_at": datetime.now(UTC).isoformat(),
        "model_sha256": sha256_file(model_file) if model_file.exists() else None,
    }
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"训练完成：{args.output}")


if __name__ == "__main__":
    main()
