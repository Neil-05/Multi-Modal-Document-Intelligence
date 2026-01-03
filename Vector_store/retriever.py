import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path, meta_path, top_k=5):
        self.index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.top_k = top_k

    def retrieve(self, query):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, self.top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results


if __name__ == "__main__":
    retriever = Retriever(
        "data/embeddings/vector.index",
        "data/embeddings/metadata.pkl",
        top_k=5
    )

    query = "What caused the slowdown in economic growth?"
    results = retriever.retrieve(query)

    print("\n🔎 Retrieved Context:\n")
    for r in results:
        print(f"- Page {r['page']} ({r['modality']}): {r['content'][:120]}...")
