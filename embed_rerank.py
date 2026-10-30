# embed_rerank.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, CHROMA_PATH, COLLECTION_NAME

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = (device == "cuda")

reranker = None
if os.path.isdir(RERANK_MODEL_NAME):
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(
            RERANK_MODEL_NAME,
            use_fp16=use_fp16,
            max_length=256,
            devices=device,
        )
        print(f"[embed_rerank] Reranker 加载成功 device={device} fp16={use_fp16}")
    except Exception as e:
        print(f"[embed_rerank] Reranker 加载失败，降级: {e}")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def get_top10_chunks(query_text: str, top_n: int = 30):
    q_vec = embed_model.encode(query_text, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=top_n)
    chunks = res["documents"][0] if res.get("documents") else []
    if not chunks:
        return []
    if reranker is None:
        return chunks[:10]
    pairs = [[query_text, c] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True, batch_size=32)
    if isinstance(scores, (int, float)):
        scores = [scores]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked[:10]]