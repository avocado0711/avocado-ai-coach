# rebuild_from_txt.py
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME, CHROMA_PATH, COLLECTION_NAME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_FILE = os.path.join(BASE_DIR, "docs", "strength_training.txt")
CHUNK_SIZE = 400     # 稍微调大，让上下文更完整
CHUNK_OVERLAP = 100  # 增加重叠，防止边界信息丢失

def split_into_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    递归字符分割器：优先按段落切，再按句号/问号/感叹号切，尽量不破坏句子
    """
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 1. 优先按双换行（段落）切
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 如果当前段落自己就超过了 size，按句子切
        if len(para) > size:
            sentences = re.split(r'(?<=[。！？!?])', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) > size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # 保留 overlap，从上一个 chunk 的尾部截取
                        current_chunk = current_chunk[-overlap:] + sent
                    else:
                        current_chunk = sent
                else:
                    current_chunk += sent
        else:
            if len(current_chunk) + len(para) > size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-overlap:] + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks


def main():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    with open(TXT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"原文总长度: {len(full_text)} 字")
    
    # 执行语义切分
    chunks = split_into_chunks(full_text)
    print(f"生成 {len(chunks)} 个高质量 chunk (平均长度 {len(full_text)//max(len(chunks),1)} 字)")

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, docs, metas = [], [], []
    for idx, chunk in enumerate(chunks):
        cid = f"doc_{idx}"
        ids.append(cid)
        docs.append(chunk)
        # 记录该 chunk 来源于哪个大段落/文件，方便后续排查
        metas.append({"doc_id": cid, "chunk_idx": idx})

    BATCH = 64
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        embs = embed_model.encode(batch, normalize_embeddings=True).tolist()
        collection.add(
            ids=ids[i:i + BATCH],
            documents=batch,
            embeddings=embs,
            metadatas=metas[i:i + BATCH],
        )
        print(f"  已写入 {i + len(batch)}/{len(docs)}")

    print("✅ 知识库重建完成！现在你可以重新运行 evaluate.py 了")


if __name__ == "__main__":
    main()
   