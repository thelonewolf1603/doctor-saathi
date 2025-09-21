# scripts/ingest_pmc.py
import os, json
from datasets import load_dataset
from tqdm import tqdm
from app.core.embeddings import EmbeddingBackend
from app.core.vectorstore import FaissStore

# config
NUM_DOCS = 1000  # for dev, ingest only a small slice

def run_ingest():
    ds = load_dataset("pmc/open_access", split="train[:{}]".format(NUM_DOCS))
    emb = EmbeddingBackend("all-MiniLM-L6-v2")
    store = FaissStore(dim=384)

    texts, metas = [], []
    for i, row in enumerate(tqdm(ds)):
        if not row.get("article"):
            continue
        text = row["article"]
        # split into chunks of ~500 chars
        chunks = [text[j:j+500] for j in range(0, len(text), 500)]
        for chunk_id, chunk in enumerate(chunks):
            texts.append(chunk)
            metas.append({
                "id": f"pmc_{i}_{chunk_id}",
                "title": row.get("title", "NA"),
                "text": chunk
            })

    embeddings = emb.embed(texts)
    store.add(embeddings, metas)
    print(f"Ingested {len(texts)} passages into FAISS.")

if __name__ == "__main__":
    run_ingest()