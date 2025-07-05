# app.py
import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from document_loaders.load_documents import load_documents_from_folder, hash_file
from embedding_docs.embed_and_store import CHROMA_PATH, INDEX_FILE
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chains.chain import build_qa_chain

load_dotenv()

DOCUMENTS_FOLDER = "data/documents"
SESSIONS_FOLDER = "data/sessions"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(SESSIONS_FOLDER, exist_ok=True)
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

EXPLANATION_TYPES = [
    "step_by_step", "analogy", "visual", "story",
    "causal", "real_life", "compare", "interactive", "socratic"
]

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_history = []
    st.session_state.chat_chain = None
    st.session_state.pending_followup = None
    st.session_state.understood_topics = []
    st.session_state.explanation_attempts = {}
    st.session_state.explanation_success = {}
    st.session_state.successful_explanation_types = set()

# --- Chat History ---
session_id = st.session_state.session_id
chat_path = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
if os.path.exists(chat_path):
    with open(chat_path, "r") as f:
        st.session_state.chat_history = [tuple(pair) for pair in json.load(f)]

# --- Sidebar: Sessions ---
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

# --- Title and Upload ---
st.title("📄 RAG Tutor with Explanation Styles and Memory")
with st.expander("➕ Upload PDFs", expanded=False):
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def embed_and_store_new_docs():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            indexed_hashes = json.load(f)
    else:
        indexed_hashes = {}

    doc_tuples = load_documents_from_folder(DOCUMENTS_FOLDER)
    new_docs = []
    for file_name, page_num, text, file_hash in doc_tuples:
        if indexed_hashes.get(file_name) != file_hash:
            new_docs.append((file_name, page_num, text, file_hash))

    if not new_docs:
        return

    st.info(f"📄 Found {len(new_docs)} new or changed pages. Embedding...")

    documents = [Document(page_content=text, metadata={"source": file_name, "page": page_num})
                 for file_name, page_num, text, _ in new_docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        vectorstore.add_documents(split_docs)
    else:
        vectorstore = Chroma.from_documents(split_docs, embedding_function, persist_directory=CHROMA_PATH)

    for file_name, _, _, file_hash in new_docs:
        indexed_hashes[file_name] = file_hash
    with open(INDEX_FILE, "w") as f:
        json.dump(indexed_hashes, f, indent=2)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(DOCUMENTS_FOLDER, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    st.success("✅ Uploaded. Now embedding...")
    embed_and_store_new_docs()
    st.success("✅ Done! You can now ask questions.")
    st.rerun()

if os.path.exists(DOCUMENTS_FOLDER):
    with st.expander("📂 View Uploaded PDFs"):
        for f in os.listdir(DOCUMENTS_FOLDER):
            st.markdown(f"- `{f}`")

@st.cache_resource
def get_vectorstore():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

vectorstore = get_vectorstore()

if st.session_state.chat_chain is None:
    st.session_state.chat_chain = build_qa_chain(vectorstore)

chat_input = st.chat_input("💬 Ask something about your documents")
query = st.session_state.pending_followup or chat_input
st.session_state.pending_followup = None
followups = []

if query:
    with st.spinner("🤖 Thinking..."):
        result = st.session_state.chat_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history[-5:],
            "understood_topics": st.session_state.understood_topics,
            "explanation_type": "default",
            "context": ""
        })

    answer = result.get("answer", "")
    if "Follow-up questions:" in answer:
        answer, raw_followups = answer.split("Follow-up questions:")
        followups = [q.strip("-• \n") for q in raw_followups.strip().split("\n") if q.strip()]

    st.session_state.chat_history.append((query, answer.strip()))
    with open(chat_path, "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)

for i, (user, bot) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(user)

    with st.chat_message("assistant"):
        st.markdown(bot)

        key_prefix = f"turn_{i}"

        if i == len(st.session_state.chat_history) - 1:
            if f"{key_prefix}_understood" not in st.session_state:
                st.markdown("**🤔 Did you understand the topic?**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Yes", key=f"{key_prefix}_yes"):
                        st.session_state[f"{key_prefix}_understood"] = True
                        topic = user.strip()
                        if topic not in st.session_state.understood_topics:
                            st.session_state.understood_topics.append(topic)
                        attempts = st.session_state.explanation_attempts.get(topic, [])
                        if attempts:
                            st.session_state.explanation_success[topic] = attempts[-1]
                            st.session_state.successful_explanation_types.add(attempts[-1])
                        st.rerun()
                with col2:
                    if st.button("👎 No, explain again", key=f"{key_prefix}_no"):
                        st.session_state[f"{key_prefix}_understood"] = False
                        st.rerun()

            elif st.session_state[f"{key_prefix}_understood"] is False:
                topic = user.strip()
                tried = st.session_state.explanation_attempts.get(topic, [])
                preferred = list(st.session_state.successful_explanation_types)
                fallback_order = preferred + [et for et in EXPLANATION_TYPES if et not in preferred]
                next_type = next((et for et in fallback_order if et not in tried), None)

                if not next_type:
                    st.warning("Tried all explanation types. Please rephrase or ask a tutor.")
                else:
                    with st.spinner(f"🔁 Re-explaining using {next_type.replace('_', ' ').title()}..."):
                        retry_result = st.session_state.chat_chain.invoke({
                            "question": user,  # use the original question, not clarified
                            "chat_history": st.session_state.chat_history[-5:],
                            "understood_topics": st.session_state.understood_topics,
                            "explanation_type": next_type,
                            "context": ""
                        })

                    retry_answer = retry_result.get("answer", "")
                    tried.append(next_type)
                    st.session_state.explanation_attempts[topic] = tried
                    st.session_state.chat_history[i] = (user, retry_answer.strip())
                    del st.session_state[f"{key_prefix}_understood"]
                    st.rerun()

            elif st.session_state[f"{key_prefix}_understood"] is True and followups:
                st.markdown("**💡 Follow-up questions:**")
                for fq in followups:
                    if st.button(fq, key=f"{key_prefix}_fq_{fq}"):
                        st.session_state.pending_followup = fq
                        st.rerun()

with st.sidebar.expander("✅ Topics You Understood"):
    for topic in st.session_state.understood_topics:
        st.markdown(f"- {topic}")

with st.sidebar.expander("🎯 Helpful Explanation Types"):
    for topic, ex_type in st.session_state.explanation_success.items():
        st.markdown(f"- **{topic}** → ✅ *{ex_type.replace('_', ' ').title()}*")

with st.sidebar.expander("🧠 Your Preferred Explanation Types"):
    for t in st.session_state.successful_explanation_types:
        st.markdown(f"- {t.replace('_', ' ').title()}")
