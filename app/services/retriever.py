# app/services/retriever.py
from app.core.embeddings import EmbeddingBackend
from app.core.vectorstore import FaissStore
import numpy as np

emb_backend = EmbeddingBackend()
# dimension depends on model; for all-MiniLM-L6-v2 it's 384
VECTOR_DIM = 384
store = FaissStore(dim=VECTOR_DIM)

def retrieve_top_k(query_text, k=5):
    qvec = emb_backend.embed([query_text])[0]
    results = store.search(qvec, k=k)[0]
    return results