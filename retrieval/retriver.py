# retrieval/retriever.py

from embed.embedder import get_embedding
from embed.vector import search_index

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[str]:
    query_vec = get_embedding(query)
    results = search_index(query_vec, top_k=top_k)
    return [chunk for chunk, _ in results]
