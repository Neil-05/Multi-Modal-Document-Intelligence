import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer





def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    return index


def save_index(index, index_path):
    faiss.write_index(index, index_path)


def save_metadata(chunks, meta_path):
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)


def build_embeddings(chunks_path, index_path, meta_path):
    chunks = load_chunks(chunks_path)
    texts = [chunk["content"] for chunk in chunks]

    embeddings = create_embeddings(texts)
    index = build_faiss_index(embeddings)

    save_index(index, index_path)
    save_metadata(chunks, meta_path)

    print("✅ Embedding & indexing complete")
    print(f"• Chunks indexed : {len(chunks)}")
    print(f"• Vector size    : {embeddings.shape[1]}")


if __name__ == "__main__":
    build_embeddings(
        chunks_path="data/processed/chunks.json",
        index_path="data/embeddings/vector.index",
        meta_path="data/embeddings/metadata.pkl"
    )
