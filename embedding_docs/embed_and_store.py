import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from document_loaders.load_documents import load_documents_from_folder
from captioning.caption_images import caption_image  # You must create this module

# Constants
CHROMA_PATH = "chroma_db"
IMAGE_CHROMA_PATH = "chroma_image_db"
DOCUMENTS_FOLDER = "data/documents"
IMAGE_FOLDER = "data/images"
INDEX_FILE = os.path.join(CHROMA_PATH, "indexed_files.json")
IMAGE_INDEX_FILE = os.path.join(IMAGE_CHROMA_PATH, "indexed_images.json")
EMBEDDING_MODEL = "thenlper/gte-small"

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def embed_new_documents():
    """Embeds new or modified documents."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    # Load existing hashes
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            indexed_hashes = json.load(f)
    else:
        indexed_hashes = {}

    # Load documents and identify new/changed ones
    doc_tuples = load_documents_from_folder(DOCUMENTS_FOLDER)
    new_documents = [
        (fname, page, text, fhash)
        for fname, page, text, fhash in doc_tuples
        if indexed_hashes.get(f"{fname}::p{page}") != fhash
    ]

    if not new_documents:
        print("✅ No new documents to embed.")
    else:
        print(f"📄 Found {len(new_documents)} new/changed document pages. Embedding...")

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

        print(f"✅ Embedded and stored {len(split_docs)} document chunks.")

    # Also embed new images
    embed_image_captions()

def embed_image_captions():
    """Embeds image captions into a separate vectorstore."""
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_CHROMA_PATH, exist_ok=True)

    if os.path.exists(IMAGE_INDEX_FILE):
        with open(IMAGE_INDEX_FILE, "r") as f:
            indexed_images = json.load(f)
    else:
        indexed_images = {}

    new_images = []
    for img_file in os.listdir(IMAGE_FOLDER):
        if img_file.endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(IMAGE_FOLDER, img_file)
            file_hash = str(os.path.getmtime(path))
            if indexed_images.get(img_file) != file_hash:
                caption = caption_image(path)
                doc = Document(page_content=caption, metadata={"image_path": path})
                new_images.append((img_file, file_hash, doc))

    if not new_images:
        print("✅ No new images to caption and embed.")
        return

    docs = [doc for _, _, doc in new_images]

    if os.path.exists(IMAGE_CHROMA_PATH) and os.listdir(IMAGE_CHROMA_PATH):
        vectorstore = Chroma(persist_directory=IMAGE_CHROMA_PATH, embedding_function=embedding_function)
        vectorstore.add_documents(docs)
    else:
        vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=IMAGE_CHROMA_PATH)

    vectorstore.persist()

    for img_file, file_hash, _ in new_images:
        indexed_images[img_file] = file_hash

    with open(IMAGE_INDEX_FILE, "w") as f:
        json.dump(indexed_images, f, indent=2)

    print(f"🖼️ Embedded {len(new_images)} new image captions.")
