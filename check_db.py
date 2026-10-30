# check_db.py
import chromadb
from config import CHROMA_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

# 拿前 5 条出来看
r = collection.get(limit=5, include=["documents"])
for i, doc_id in enumerate(r["ids"]):
    print(f"\n--- {doc_id} ---")
    print(r["documents"][i][:150] + "...")