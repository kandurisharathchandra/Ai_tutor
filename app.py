import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chains.chain import build_qa_chain
from embedding_docs.embed_and_store import CHROMA_PATH, INDEX_FILE, embed_new_documents, IMAGE_CHROMA_PATH

load_dotenv()

DOCUMENTS_FOLDER = "data/documents"
SESSIONS_FOLDER = "data/sessions"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(SESSIONS_FOLDER, exist_ok=True)
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_history = []
    st.session_state.chat_chain = None
    st.session_state.pending_followup = None

# --- Chat History File ---
session_id = st.session_state.session_id
chat_path = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
if os.path.exists(chat_path):
    with open(chat_path, "r") as f:
        st.session_state.chat_history = [tuple(pair) for pair in json.load(f)]

# --- Sidebar Sessions ---
st.sidebar.title("🗂️ Saved Sessions")
sessions = sorted([f for f in os.listdir(SESSIONS_FOLDER) if f.endswith(".json")], reverse=True)
selected_session = st.sidebar.selectbox("Select session", sessions, index=0 if sessions else None)
if selected_session:
    with open(os.path.join(SESSIONS_FOLDER, selected_session)) as f:
        session_data = json.load(f)
    st.sidebar.markdown(f"### 💬 {selected_session}")
    for i, (user, bot) in enumerate(reversed(session_data), 1):
        with st.sidebar.expander(f"{i}. {user[:30]}..."):
            st.markdown(f"**🧑 You:** {user}")
            st.markdown(f"**🤖 AI:** {bot}")

if st.sidebar.button("🗑️ Delete This Session"):
    os.remove(os.path.join(SESSIONS_FOLDER, selected_session))
    st.sidebar.success("Deleted.")
    st.rerun()

# --- Title & Upload ---
st.title("📄 RAG App with Image-Aware QA")
with st.expander("➕ Upload PDFs", expanded=False):
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# --- Embedding ---
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(DOCUMENTS_FOLDER, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    st.success("✅ Uploaded. Now embedding...")
    embed_new_documents()  # includes both doc and image embedding
    st.success("✅ Done! You can now ask questions.")
    st.rerun()

# --- View Uploaded Files ---
if os.path.exists(DOCUMENTS_FOLDER):
    with st.expander("📂 View Uploaded PDFs"):
        for f in os.listdir(DOCUMENTS_FOLDER):
            st.markdown(f"- `{f}`")

# --- Cached Vectorstore Load ---
@st.cache_resource
def get_vectorstore():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@st.cache_resource
def get_image_vectorstore():
    return Chroma(persist_directory=IMAGE_CHROMA_PATH, embedding_function=embedding_function)

vectorstore = get_vectorstore()
image_vectorstore = get_image_vectorstore()

# --- Build QA Chain ---
if st.session_state.chat_chain is None:
    st.session_state.chat_chain = build_qa_chain(vectorstore)

# --- Chat Input ---
chat_input = st.chat_input("💬 Ask something about your documents")
query = st.session_state.pending_followup or chat_input
st.session_state.pending_followup = None
followups = []

if query:
    with st.spinner("🤖 Thinking..."):
        result = st.session_state.chat_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history[-5:]
        })

    answer = result.get("answer", "")
    
    # Extract follow-up questions if present
    if "Follow-up questions:" in answer:
        answer, raw_followups = answer.split("Follow-up questions:")
        followups = [q.strip("-• \n") for q in raw_followups.strip().split("\n") if q.strip()]

    # --- Retrieve Relevant Image (Top 1) ---
    relevant_imgs = image_vectorstore.similarity_search(query, k=1)
    if relevant_imgs:
        img_meta = relevant_imgs[0].metadata
        image_path = img_meta.get("image_path")
        if image_path and os.path.exists(image_path):
            answer += f"\n\n**📷 Relevant Image:**"
            st.session_state.chat_history.append((query, answer.strip()))
        else:
            st.session_state.chat_history.append((query, answer.strip()))
    else:
        st.session_state.chat_history.append((query, answer.strip()))

    with open(chat_path, "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)

# --- Show Chat ---
for user, bot in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)

        # Show image if caption present in bot text
        if "**📷 Relevant Image:**" in bot:
            # Match with image caption again
            last_img = image_vectorstore.similarity_search(user, k=1)
            if last_img:
                img_path = last_img[0].metadata.get("image_path")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

# --- Show Follow-up ---
if followups:
    st.markdown("**💡 Follow-up questions:**")
    for fq in followups:
        st.markdown(f"- {fq}")
