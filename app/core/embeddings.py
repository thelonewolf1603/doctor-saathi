# app/core/embeddings.py
from sentence_transformers import SentenceTransformer
import os
import numpy as np

class EmbeddingBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype(np.float32)