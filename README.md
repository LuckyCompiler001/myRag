# 🧠 Retrieval-Augmented Generation (RAG) Pipeline Prototype  
**Summer 2025 Project by LuckyCompiler001**

This repository contains a functional prototype of a **Retrieval-Augmented Generation (RAG)** system, which combines vector-based document retrieval with large language model (LLM) response generation. It’s designed as a modular, end-to-end pipeline to process and query your own document corpus.

---

## 📁 Project Structure

```
├── config.py               # Configuration file (API keys, model settings)
├── main.py                 # Entry point to run the RAG pipeline
├── ingest/                 # Load and preprocess raw documents
├── embed/                  # Generate and store vector embeddings
├── retrieval/              # Similarity-based document retrieval logic
├── llm_generation/         # Prompt construction and LLM response handling
├── helper_utilities/       # Shared utilities (I/O, logging, formatting)
├── data/raw/               # Input files (PDFs, text, etc.)
├── outputs/                # Logs and generated results
├── LICENSE                 # License information
└── README.md               # Project documentation
```

---

## ⚙️ Setup
Make sure to install the required dependencies. Populate your `requirements.txt` with:
openai
faiss-cpu
python-dotenv
Then, install them using:
bash
pip install -r requirements.txt

Also, ensure your `.env` file includes necessary credentials such as your OpenAI API key.

## 🚀 Usage

Run the pipeline using the `main.py` entry point:
### 1. Ingest & Embed Documents

bash
python main.py

When prompted, choose:
```
> build
```

This will load, preprocess, and embed your source documents.

### 2. Query the System

bash
python main.py

Then choose:
```
> query
```
Enter your question when prompted. For example:
```
> What are the main ideas from file X?
```
---
## 📌 Notes

- The current version is a prototype and may require adaptation for large-scale or production use.
- All embeddings are stored locally; cloud-based vector databases can be integrated as needed.
- The system is model-agnostic—swap out the LLM or embedding model via `config.py`.
---
Feel free to extend, modify, or contribute to improve this RAG system. Enjoy exploring your documents with AI!
