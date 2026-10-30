# evaluate.py
import json
import os
import sys
import time
from collections import defaultdict

import torch
import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# ★ 改动 1：import 加上 RERANK_MODEL_NAME
from config import EMBED_MODEL_NAME, RERANK_MODEL_NAME, CHROMA_PATH, COLLECTION_NAME

# 强制走镜像并且离线
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

# ==================== 新增：自动GPU检测 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = (DEVICE == "cuda")  # 有N卡自动开半精度加速，CPU自动关闭不报错
print(f"✅ 当前运行设备: {DEVICE} | 半精度(fp16): {USE_FP16}\n")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(BASE_DIR, "eval_set.json1")
TOP_K = 5

# ==================== 初始化模型与数据 ====================
# 修复：Embedding模型指定运行设备
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
# ★ 改动 2：Rerank 改用本地配置路径，不再联网
reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=USE_FP16, device=DEVICE)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

# 一次性把全量数据拉出来，用于构建 BM25 索引
_all_data = collection.get(include=["documents", "metadatas"])
all_ids = _all_data["ids"]
all_docs = _all_data["documents"]
# 构建 id -> doc 的映射
id_to_doc = dict(zip(all_ids, all_docs))

# 构建 BM25 索引 (jieba 分词)
tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)


def load_eval_set():
    items = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                it = json.loads(line)
                if "gold_doc_ids" not in it:
                    it["gold_doc_ids"] = [it["gold_doc_id"]]
                items.append(it)
    return items


def _hit_and_rank(ids, gold_set, k):
    for rank, doc_id in enumerate(ids[:k], start=1):
        if doc_id in gold_set:
            return True, rank
    return False, 0


# ==================== 混合检索 (核心修复) ====================
def hybrid_retrieve(query, top_k_embed=50, top_k_rerank=10):
    # 1. 向量召回
    q_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=top_k_embed)
    embed_ids = res["ids"][0] if res.get("ids") and res["ids"][0] else []

    # 2. BM25 召回
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    # 取分数最高的 top_k_embed 个索引
    bm25_top_idx = bm25_scores.argsort()[-top_k_embed:][::-1]
    bm25_ids = [all_ids[i] for i in bm25_top_idx]

    # 3. RRF 融合排名
    rrf_scores = defaultdict(float)
    for rank, doc_id in enumerate(embed_ids):
        rrf_scores[doc_id] += 1.0 / (rank + 60)
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] += 1.0 / (rank + 60)

    # 按融合分数降序，取 Top 30 个候选进入 Reranker
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    candidate_ids = [cid for cid, _ in sorted_candidates]
    candidate_docs = [id_to_doc[cid] for cid in candidate_ids]

    # 4. Reranker 精排（修复：加批处理加速）
    pairs = [[query, doc] for doc in candidate_docs]
    rerank_scores = reranker.compute_score(pairs, normalize=True, batch_size=16)

    # 结合原始 id 和重排分数排序
    reranked = []
    for cid, score in zip(candidate_ids, rerank_scores):
        reranked.append((cid, score))
    reranked.sort(key=lambda x: x[1], reverse=True)

    return [cid for cid, _ in reranked[:top_k_rerank]]


# ==================== 评估函数 ====================
def eval_embedding_only(items, k=TOP_K):
    hits, mrr = 0, 0.0
    n_valid = 0
    for it in items:
        gold = set(it["gold_doc_ids"])
        if not gold: continue
        n_valid += 1
        q_vec = embed_model.encode(it["query"], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=[q_vec], n_results=k)
        ids = res["ids"][0] if res.get("ids") and res["ids"][0] else []
        hit, rank = _hit_and_rank(ids, gold, k)
        if hit:
            hits += 1
            mrr += 1.0 / rank
    return hits / max(n_valid, 1), mrr / max(n_valid, 1)


def eval_with_rerank(items, k=TOP_K):
    hits, mrr = 0, 0.0
    n_valid = 0
    for it in items:
        q = it["query"]
        gold = set(it["gold_doc_ids"])
        if not gold: continue
        n_valid += 1

        ranked_ids = hybrid_retrieve(q, top_k_embed=50, top_k_rerank=k)
        hit, rank = _hit_and_rank(ranked_ids, gold, k)
        if hit:
            hits += 1
            mrr += 1.0 / rank
    return hits / max(n_valid, 1), mrr / max(n_valid, 1)


# ==================== 诊断 (原样保留) ====================
def diag_basic(collection, items):
    print("=== 诊断 0：基础信息 ===")
    print(f"  库总条数: {collection.count()}")
    r = collection.get(limit=5, include=["metadatas", "documents"])
    print(f"  前 5 个 id: {r['ids']}")
    print(f"  前 5 个 doc 前 30 字: {[d[:30] for d in r['documents']]}\n")


def diag_gold_count(items):
    print("=== 诊断 4：每个 query 的 gold 数量 ===")
    n_rel = [len(it["gold_doc_ids"]) for it in items]
    print(f"  min={min(n_rel)}, max={max(n_rel)}, mean={sum(n_rel)/len(n_rel):.1f}\n")


def diag_gold_exists(collection, items):
    print("=== 诊断 1：gold 是否在库中（去重后） ===")
    all_gold = [gid for it in items for gid in it["gold_doc_ids"]]
    uniq = list(set(all_gold))
    found = set()
    for i in range(0, len(uniq), 100):
        r = collection.get(ids=uniq[i:i+100])
        found.update(r.get("ids", []))
    missing = set(uniq) - found
    print(f"  gold 唯一数: {len(uniq)}，能找到: {len(found)}，缺失: {len(missing)}\n")


def diag_top5_detail(embed_model, collection, items):
    print("=== 诊断 2：前 5 条 query 检索详情 (纯 Embedding) ===")
    for it in items[:5]:
        q = it["query"]
        gold = set(it["gold_doc_ids"])
        q_vec = embed_model.encode(q, normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=[q_vec], n_results=10)
        ids = res["ids"][0]
        docs = res["documents"][0]
        hit_rank = next((i+1 for i, d in enumerate(ids) if d in gold), None)
        print(f"  [{q}]\n    gold: {list(gold)}\n    top10 ids: {ids}\n    top1 文本: {docs[0][:60]}...\n    命中位置: {hit_rank}")
    print()


def diag_recall_curve(embed_model, collection, items):
    print("=== 诊断 3：Recall@K (纯 embedding) ===")
    Ks = [5, 10, 20, 50, 100, 200]
    hit_cnt = {k: 0 for k in Ks}
    mrr_sum = {k: 0.0 for k in Ks}
    n_valid = 0
    for it in items:
        gold = set(it["gold_doc_ids"])
        if not gold: continue
        n_valid += 1
        q_vec = embed_model.encode(it["query"], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=[q_vec], n_results=max(Ks))
        ids = res["ids"][0]
        first_rank = {}
        for rank, d in enumerate(ids, start=1):
            if d in gold and d not in first_rank:
                first_rank[d] = rank
        for k in Ks:
            best = min((r for r in first_rank.values() if r <= k), default=None)
            if best is not None:
                hit_cnt[k] += 1
                mrr_sum[k] += 1.0 / best
    for k in Ks:
        print(f"  Recall@{k:<3d} = {hit_cnt[k]/n_valid:.4f}   MRR@{k:<3d} = {mrr_sum[k]/n_valid:.4f}")
    print(f"  (有效 query 数 = {n_valid})\n")


def main():
    if not os.path.exists(EVAL_FILE):
        print(f"缺少评估集文件：{EVAL_FILE}"); sys.exit(1)

    items = load_eval_set()
    print(f"样本数: {len(items)}\n")

    diag_basic(collection, items)
    diag_gold_count(items)
    diag_gold_exists(collection, items)
    diag_top5_detail(embed_model, collection, items)
    diag_recall_curve(embed_model, collection, items)

    print("=== 纯 embedding ===")
    t0 = time.time()
    h1, m1 = eval_embedding_only(items)
    print(f"Hit@{TOP_K}: {h1:.4f}  MRR@{TOP_K}: {m1:.4f}  耗时 {time.time()-t0:.1f}s\n")

    print("=== embedding + hybrid + rerank ===")
    t0 = time.time()
    h2, m2 = eval_with_rerank(items)
    print(f"Hit@{TOP_K}: {h2:.4f}  MRR@{TOP_K}: {m2:.4f}  耗时 {time.time()-t0:.1f}s\n")

    print("=== 提升 ===")
    print(f"Hit@{TOP_K} 提升: {(h2-h1)*100:+.2f} 个百分点")
    print(f"MRR@{TOP_K} 提升: {(m2-m1)*100:+.2f} 个百分点")


if __name__ == "__main__":
    main()