# embed/vector_store.py

# pip install faiss-cpu
import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple

INDEX_PATH = "outputs/cache/faiss.index"
META_PATH = "outputs/cache/chunk_meta.pkl"  # stores the original text chunks

def create_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    if not embeddings:
        raise ValueError("No embeddings provided. Check document content.")

    vector_dim = len(embeddings[0])
    print(f"[Vector Store] Creating index with dimension: {vector_dim}")

    index = faiss.IndexFlatL2(vector_dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def save_index(index: faiss.IndexFlatL2):
    faiss.write_index(index, INDEX_PATH)

def load_index() -> faiss.IndexFlatL2:
    return faiss.read_index(INDEX_PATH)

def save_chunk_metadata(chunks: List[str]):
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_chunk_metadata() -> List[str]:
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

def search_index(query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
    index = load_index()
    chunks = load_chunk_metadata()

    query = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append((chunks[idx], dist))
    return results
