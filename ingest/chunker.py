from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_documents(docs: List[str], chunk_size=500, overlap=100) -> List[str]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks