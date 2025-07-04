import os
import hashlib
from PyPDF2 import PdfReader

def hash_file(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def load_documents_from_folder(folder_path: str):
    """
    Loads all PDFs in a folder and extracts text (no OCR).
    Returns a list of tuples: (file_name, page_number, text, hash)
    """
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(file_path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text = text.strip()
                    if not text:
                        continue
                    file_hash = hash_file(text)
                    all_docs.append((filename, page_num + 1, text, file_hash))
            except Exception as e:
                print(f"⚠️ Could not read {filename}: {e}")
    
    return all_docs
