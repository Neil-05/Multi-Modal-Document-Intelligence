import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

MODEL_NAME = "all-MiniLM-L6-v2"

def build_embeddings(chunks_path, index_path, meta_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["content"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)
