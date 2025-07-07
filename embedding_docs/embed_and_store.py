# embedding_docs/embed_and_store.py

import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from document_loaders.load_documents import load_documents_from_folder

# Constants
CHROMA_PATH = "chroma_db"
DOCUMENTS_FOLDER = "data/documents"
INDEX_FILE = os.path.join(CHROMA_PATH, "indexed_files.json")

# 🔄 Updated embedding model: BAAI/bge-small-en-v1.5
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},           # Change to "cuda" if you want GPU
    encode_kwargs={"normalize_embeddings": True}  # Required for BGE performance
)


def embed_new_documents():
    """
    Load documents, check for new/changed content, split and embed them,
    and persist into Chroma vectorstore.
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            indexed_hashes = json.load(f)
    else:
        indexed_hashes = {}

    doc_tuples = load_documents_from_folder(DOCUMENTS_FOLDER)
    new_documents = [
        (fname, page, text, fhash)
        for fname, page, text, fhash in doc_tuples
        if indexed_hashes.get(f"{fname}::p{page}") != fhash
    ]

    if not new_documents:
        print("✅ No new documents to embed.")
        return

    print(f"📄 Found {len(new_documents)} new/changed pages. Embedding...")

    documents = [
        Document(page_content=text, metadata={"source": fname, "page": page})
        for fname, page, text, _ in new_documents
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        vectorstore.add_documents(split_docs)
    else:
        vectorstore = Chroma.from_documents(split_docs, embedding_function, persist_directory=CHROMA_PATH)

    vectorstore.persist()

    for fname, page, _, fhash in new_documents:
        indexed_hashes[f"{fname}::p{page}"] = fhash
    with open(INDEX_FILE, "w") as f:
        json.dump(indexed_hashes, f, indent=2)

    print(f"✅ Embedded and stored {len(split_docs)} chunks.")
