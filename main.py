import sys
import os   
import logging
import os
import logging
from ingest.loader import load_documents_from_folder
from ingest.chunker import chunk_documents
from embed.embedder import get_embedding
from embed.vector import create_faiss_index, save_index, save_chunk_metadata
from retrieval.retriver import retrieve_relevant_chunks
from llm_generation.llm_generator import generate_answer





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

import os

def initializer():
    base_dirs = {
        "data": ["raw", "processed"],
        "outputs": ["cache", "logs"]
    }

    for base, subdirs in base_dirs.items():
        for sub in subdirs:
            path = os.path.join(base, sub)
            if not os.path.exists(path):
                os.makedirs(path)

    # Also ensure the top-level 'outputs' directory itself exists
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
        # === Logging Setup ===
    log_dir = os.path.join("outputs", "logs")
    log_file_name = 'logfile.log'
    log_path = os.path.join(log_dir, log_file_name)


    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')
    
    logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(message)s - %(filename)s - funcname: %(funcName)s - %(asctime)s - %(levelname)s'
        )

    logging.info("RAG system started. Logging initialized.")

import shutil

def clean_processed_and_cache():
    target_dirs = [
        os.path.join("data", "processed"),
        os.path.join("outputs", "cache")
    ]

    for dir_path in target_dirs:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logging.info(f"Deleted directory: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}. Reason: {e}")



if __name__ == "__main__":
    initializer()
    mode = input("Choose mode [build/query]: ").strip().lower()

    if mode == "build":
        build_index(os.path.join("data", "raw"))
        print("Index building complete. You can now query the system.")
        logging.info("Index building complete. You can now query the system.")
    elif mode == "query":
        while True:
            flag = input("you wish to quit [Y/N]: ").strip().lower()
            if flag == 'y':
                logging.info("Exiting query mode.")
                print("Exiting query mode.")
                clean = input("Do you want to clean processed and cache directories? [Y/N]: ").strip().lower()
                if clean == 'y':
                    clean_processed_and_cache()
                    logging.info("Processed and cache directories cleaned.")
                    print("Processed and cache directories cleaned.")
                break
            elif flag != 'n':
                logging.warning("Invalid input. Please enter 'Y' or 'N'.")
                print("Invalid input. Please enter 'Y' or 'N'.")
                continue
            query = input("Enter your question: ")
            query_rag(query)
            logging.info(f"User query: {query}")
    else:
        logging.warning("Invalid mode selected. Use 'build' or 'query'.")
        print("Unknown mode. Use 'build' or 'query'.")
