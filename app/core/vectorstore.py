# app/core/vectorstore.py
import faiss
import numpy as np
import os
import pickle

class FaissStore:
    def __init__(self, dim, index_path="data/index/faiss.index", meta_path="data/index/meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)  # inner product (use normalized vectors)
            self.meta = []

    def add(self, embeddings: np.ndarray, metadatas: list):
        # normalize for IP
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = embeddings / norms
        self.index.add(vecs)
        self.meta.extend(metadatas)
        self._save()

    def search(self, query_vec: np.ndarray, k=5):
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        # normalize query
        qn = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        D, I = self.index.search(qn, k)
        results = []
        for dist_list, idx_list in zip(D, I):
            row = []
            for dist, idx in zip(dist_list, idx_list):
                if idx < 0 or idx >= len(self.meta): 
                    continue
                md = self.meta[idx]
                row.append({"score": float(dist), "metadata": md})
            results.append(row)
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)