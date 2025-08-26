from typing import List

# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = min(start + chunk_size, len(text))
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks
# advanced_chunking.py

import re
import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from config import client, LLM_MODEL
from embed.embedder import get_embedding


def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 100,
    mode: str = "fixed"
) -> List[str]:
    """
    Enhanced chunking function supporting multiple strategies.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        mode: Chunking mode - "fixed", "semantic", "recursive", or "agentic"
    
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    if mode == "fixed":
        return _chunk_fixed(text, chunk_size, overlap)
    elif mode == "semantic":
        return _chunk_semantic(text, chunk_size)
    elif mode == "recursive":
        return _chunk_recursive(text, chunk_size, overlap)
    elif mode == "agentic":
        return _chunk_agentic(text, chunk_size)
    else:
        # Default to original method if unknown mode
        return _chunk_fixed(text, chunk_size, overlap)


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Original fixed-size chunking method."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _chunk_semantic(text: str, chunk_size: int) -> List[str]:
    """Semantic chunking using embedding similarity."""
    def split_into_sentences(text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return [text]
    
    # Get embeddings for sentences
    embeddings = []
    for sentence in sentences:
        embedding = get_embedding(sentence)
        if embedding:
            embeddings.append(embedding)
        else:
            embeddings.append([0.0] * 1536)  # Fallback
    
    if len(embeddings) < 2:
        return [text]
    
    # Calculate semantic distances
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    
    # Find breakpoints using 95th percentile threshold
    breakpoint_threshold = np.percentile(distances, 95)
    breakpoints = [i for i, distance in enumerate(distances) if distance > breakpoint_threshold]
    
    # Create chunks
    chunks = []
    start_idx = 0
    
    for breakpoint in breakpoints:
        chunk_sentences = sentences[start_idx:breakpoint + 1]
        chunk_text = ' '.join(chunk_sentences)
        
        # Respect chunk_size limit
        if len(chunk_text) > chunk_size * 1.5:  # Allow 50% flexibility
            # Split large semantic chunk further
            sub_chunks = _chunk_fixed(chunk_text, chunk_size, 50)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk_text)
        
        start_idx = breakpoint + 1
    
    # Handle remaining sentences
    if start_idx < len(sentences):
        final_chunk = ' '.join(sentences[start_idx:])
        if len(final_chunk) > chunk_size * 1.5:
            sub_chunks = _chunk_fixed(final_chunk, chunk_size, 50)
            chunks.extend(sub_chunks)
        else:
            chunks.append(final_chunk)
    
    return [chunk for chunk in chunks if chunk.strip()]


def _chunk_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Recursive chunking respecting document structure."""
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    
    def recursive_split(text: str, separators: List[str]) -> List[str]:
        if not text.strip() or len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        if not separators:
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)
        
        # Try to merge splits into chunks
        chunks = []
        current_chunk = ""
        
        for split in splits:
            potential_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single split is too large, recursively split it
                if len(split) > chunk_size:
                    sub_chunks = recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    chunks = recursive_split(text, separators)
    
    # Add overlap
    if len(chunks) <= 1:
        return chunks
    
    overlapped_chunks = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        current_chunk = chunks[i]
        
        if len(prev_chunk) > overlap:
            overlap_text = prev_chunk[-overlap:]
            word_boundary = overlap_text.rfind(' ')
            if word_boundary > 0:
                overlap_text = overlap_text[word_boundary+1:]
            
            overlapped_chunk = overlap_text + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        else:
            overlapped_chunks.append(current_chunk)
    
    return overlapped_chunks


def _chunk_agentic(text: str, chunk_size: int) -> List[str]:
    """Agentic chunking using LLM intelligence."""
    if len(text) <= chunk_size * 1.2:  # Allow some flexibility
        return [text]
    
    # Create propositions
    proposition_prompt = f"""
    Break down the following text into standalone propositions. Each proposition should be a complete, self-contained statement.

    Text: {text}

    Return only the propositions, one per line, without numbering.
    """
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at breaking down text into standalone propositions."},
                {"role": "user", "content": proposition_prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        propositions = response.choices[0].message.content.strip().split('\n')
        propositions = [prop.strip() for prop in propositions if prop.strip()]
        
    except Exception as e:
        print(f"Error creating propositions: {e}")
        # Fallback to sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        propositions = [s.strip() for s in sentences if s.strip()]
    
    if not propositions:
        return [text]
    
    # Group propositions
    indexed_props = [f"{i}: {prop}" for i, prop in enumerate(propositions)]
    props_text = '\n'.join(indexed_props)
    
    grouping_prompt = f"""
    Group these propositions into coherent chunks of roughly {chunk_size} characters each:

    {props_text}

    Return as: Group1: 0,1,3 | Group2: 2,4,5 | Group3: 6,7
    """
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at grouping related information."},
                {"role": "user", "content": grouping_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        grouping = response.choices[0].message.content.strip()
        
        # Parse grouping
        chunks = []
        group_parts = grouping.split('|')
        
        for group_part in group_parts:
            if ':' in group_part:
                indices_str = group_part.split(':')[1].strip()
                try:
                    indices = [int(i.strip()) for i in indices_str.split(',')]
                    group_props = [propositions[i] for i in indices if i < len(propositions)]
                    if group_props:
                        chunk_text = ' '.join(group_props)
                        chunks.append(chunk_text)
                except (ValueError, IndexError):
                    continue
        
        return chunks if chunks else [text]
        
    except Exception as e:
        print(f"Error grouping propositions: {e}")
        # Fallback: simple sequential grouping
        chunks = []
        current_chunk = ""
        
        for prop in propositions:
            if len(current_chunk + " " + prop) <= chunk_size * 1.2:
                current_chunk = current_chunk + " " + prop if current_chunk else prop
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = prop
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


def chunk_documents(docs: List[str], chunk_size: int = 500, overlap: int = 100, mode: str = "fixed") -> List[str]:
    """Chunk multiple documents using the specified mode."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_size, overlap, mode)
        all_chunks.extend(chunks)
    return all_chunks

def chunk_documents(docs: List[str], chunk_size=500, overlap=100) -> List[str]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks