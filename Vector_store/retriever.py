import faiss
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, index_path, meta_path, top_k=5):
        self.top_k = top_k
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        if not self.index_path.exists() or not self.meta_path.exists():
            from Vector_store.build_index import build_embeddings
            build_embeddings(
                chunks_path="data/processed/chunks.json",
                index_path=self.index_path,
                meta_path=self.meta_path
            )

        self.index = faiss.read_index(str(self.index_path))

        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, query):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, self.top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
