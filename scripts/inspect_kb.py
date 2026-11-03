"""Inspect the current Chroma collection."""

from _bootstrap import PROJECT_ROOT  # noqa: F401

from app.embed_rerank import collection, retrieval_status

data = collection.get(limit=5, include=["documents", "metadatas"])
print(retrieval_status())
for index, doc_id in enumerate(data["ids"]):
    print(f"\n--- {doc_id} ---")
    print(data["metadatas"][index])
    print((data["documents"][index] or "")[:150] + "...")
