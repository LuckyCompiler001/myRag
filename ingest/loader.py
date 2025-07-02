import os
import fitz  # PyMuPDF
# pip install PyMuPDF if needed 
from typing import List

def load_txt_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf_file(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_documents_from_folder(folder_path: str) -> List[str]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt") or filename.endswith(".md"):
            documents.append(load_txt_file(file_path))
        elif filename.endswith(".pdf"):
            documents.append(load_pdf_file(file_path))
    return documents