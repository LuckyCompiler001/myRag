import sys
import os   
import logging

def func0()->None:
    # this is a test function
    print("This is a test function in main.py")
    # in the case you want to test the library. like loggging
    log_dir = './output'
    log_file_name = 'logfile.log'
    log_path = os.path.join(log_dir, log_file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')  # Create empty file
        
    logging.basicConfig(filename ='./output/logfile.log', 
                        level = logging.INFO,
                        format ='%(message)s - %(filename)s - funcname: %(funcName)s - %(asctime)s - %(levelname)s')
    logging.info("This will be saved in the log file. And this is a test log message to track our work. hello from main.py. you will be able to see time, level, message and file and function name")

    # you can also test and expand this function as you like :)

# main.py

# main.py

import os
import logging
from ingest.loader import load_documents_from_folder
from ingest.chunker import chunk_documents
from embed.embedder import get_embedding
from embed.vector_store import create_faiss_index, save_index, save_chunk_metadata
from retrieval.retriever import retrieve_relevant_chunks
from generation.llm_generator import generate_answer

# === Logging Setup ===
log_dir = './output'
log_file_name = 'logfile.log'
log_path = os.path.join(log_dir, log_file_name)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write('')

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(message)s - %(filename)s - funcname: %(funcName)s - %(asctime)s - %(levelname)s'
)

logging.info("RAG system started. Logging initialized.")

def build_index(doc_folder: str):
    logging.info("Loading documents...")
    docs = load_documents_from_folder(doc_folder)

    logging.info("Chunking documents...")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)

    logging.info(f"Embedding {len(chunks)} chunks...")
    embeddings = [get_embedding(chunk) for chunk in chunks]

    logging.info("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    logging.info("Saving index and metadata...")
    save_index(index)
    save_chunk_metadata(chunks)

    logging.info("Index building complete.")

def query_rag(query: str):
    logging.info(f"Retrieving relevant chunks for query: {query}")
    context_chunks = retrieve_relevant_chunks(query, top_k=5)
    context = "\n\n".join(context_chunks)

    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""
    logging.info("Sending prompt to LLM...")
    answer = generate_answer(prompt)

    logging.info("Received answer from LLM.")
    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    func0()  # Call the test function
    mode = input("Choose mode [build/query]: ").strip().lower()

    if mode == "build":
        build_index("data/raw")
    elif mode == "query":
        query = input("Enter your question: ")
        query_rag(query)
    else:
        logging.warning("Invalid mode selected. Use 'build' or 'query'.")
        print("Unknown mode. Use 'build' or 'query'.")
