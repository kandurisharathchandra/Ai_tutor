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

EXPLANATION_TYPES = ["analogy", "story", "real_life", "causal", "compare"]

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_history = []
    st.session_state.chat_chain = None
    st.session_state.pending_followup = None
    st.session_state.understood_topics = []
    st.session_state.explanation_attempts = {}
    st.session_state.explanation_success = {}
    st.session_state.successful_explanation_types = set()
    st.session_state.last_explanation_type = None
    st.session_state.selected_explanation_type = "step_by_step"

session_id = st.session_state.session_id
chat_path = os.path.join(SESSIONS_FOLDER, f"{session_id}.json")
memory_path = os.path.join(SESSIONS_FOLDER, f"{session_id}_memory.json")

if os.path.exists(chat_path):
    with open(chat_path, "r") as f:
        st.session_state.chat_history = [tuple(pair) for pair in json.load(f)]
if os.path.exists(memory_path):
    with open(memory_path, "r") as f:
        memory = json.load(f)
        st.session_state.understood_topics = memory.get("understood_topics", [])
        st.session_state.explanation_success = memory.get("explanation_success", {})
        st.session_state.successful_explanation_types = set(memory.get("successful_explanation_types", []))

st.sidebar.title("🗂️ Session Navigator")
sessions = sorted([f for f in os.listdir(SESSIONS_FOLDER) if f.endswith(".json") and not f.endswith("_memory.json")], reverse=True)

selected_session = st.sidebar.selectbox("📑 Select a past session", sessions, index=0 if sessions else None)

if selected_session:
    st.sidebar.markdown(f"### 💬 {selected_session}")
    mem_file = selected_session.replace(".json", "_memory.json")
    mem_path = os.path.join(SESSIONS_FOLDER, mem_file)

    with st.sidebar.expander("📋 View Questions", expanded=False):
        session_path = os.path.join(SESSIONS_FOLDER, selected_session)
        try:
            with open(session_path, "r") as f:
                session_chat = json.load(f)
            for i, (q, a) in enumerate(session_chat):
                with st.expander(f"{i+1}. {q}", expanded=False):
                    st.markdown(a)
                    if st.button("↩️ Re-ask this", key=f"reask_{i}"):
                        st.session_state.chat_input = q
                        st.session_state.pending_followup = q
                        st.session_state.chat_history.append((q, ""))
                        st.session_state.force_reanswer = True
                        st.session_state.reask_from_button = True
                        st.rerun()
        except Exception as e:
            st.warning(f"Error loading session: {e}")

    if os.path.exists(mem_path):
        with open(mem_path) as f:
            past_mem = json.load(f)
        st.sidebar.markdown("#### ✅ Topics Understood")
        for t in past_mem.get("understood_topics", []):
            st.sidebar.markdown(f"- {t}")

    if st.sidebar.button("🗑️ Delete This Session"):
        os.remove(os.path.join(SESSIONS_FOLDER, selected_session))
        if os.path.exists(mem_path):
            os.remove(mem_path)
        st.sidebar.success("Deleted.")
        st.rerun()

# Single explanation type selector
explanation_type_changed = False
selected_type = st.selectbox(
    "🎯 Choose explanation style for this question",
    ["step_by_step"] + EXPLANATION_TYPES,
    index=( ["step_by_step"] + EXPLANATION_TYPES ).index(st.session_state.selected_explanation_type),
    key="explanation_type_selector"
)
if selected_type != st.session_state.selected_explanation_type:
    st.session_state.selected_explanation_type = selected_type
    explanation_type_changed = True

st.title("✨ Super Personalized AI Tutor")
with st.expander("➕ Upload PDFs", expanded=False):
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_vectorstore():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

vectorstore = get_vectorstore()

if st.session_state.chat_chain is None:
    st.session_state.chat_chain = build_qa_chain(vectorstore)

def embed_and_store_new_docs():
    indexed_hashes = {}
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            indexed_hashes = json.load(f)

    doc_tuples = load_documents_from_folder(DOCUMENTS_FOLDER)
    new_docs = [(f, p, t, h) for f, p, t, h in doc_tuples if indexed_hashes.get(f) != h]

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

chat_input = st.chat_input("💬 Ask something about your documents")

last_turn = st.session_state.chat_history[-1] if st.session_state.chat_history else None
last_question = last_turn[0] if last_turn else None
query = st.session_state.pending_followup or chat_input
st.session_state.pending_followup = None
followups = []

should_reanswer = False
if query and query != last_question:
    should_reanswer = True
elif explanation_type_changed and last_question:
    should_reanswer = True

if should_reanswer:
    with st.spinner(f"🤖 Explaining using {st.session_state.selected_explanation_type.replace('_', ' ').title()}..."):
        result = st.session_state.chat_chain.invoke({
            "question": query or last_question,
            "chat_history": st.session_state.chat_history[-5:],
            "understood_topics": st.session_state.understood_topics,
            "explanation_type": st.session_state.selected_explanation_type
        })

    new_answer = result.get("answer", "")
    if "Follow-up questions:" in new_answer:
        new_answer, raw_followups = new_answer.split("Follow-up questions:")
        followups = [q.strip("-• \n") for q in raw_followups.strip().split("\n") if q.strip()]

    if query:
        st.session_state.chat_history.append((query, new_answer.strip()))
    else:
        st.session_state.chat_history[-1] = (last_question, new_answer.strip())

    st.session_state.last_explanation_type = st.session_state.selected_explanation_type

    with open(chat_path, "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)
    with open(memory_path, "w") as f:
        json.dump({
            "understood_topics": st.session_state.understood_topics,
            "explanation_success": st.session_state.explanation_success,
            "successful_explanation_types": list(st.session_state.successful_explanation_types)
        }, f, indent=2)

    st.rerun()

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

                next_type = st.session_state.selected_explanation_type if st.session_state.selected_explanation_type not in tried else None

                if not next_type:
                    next_type = next((et for et in fallback_order if et not in tried), None)

                if not next_type:
                    st.warning("Tried all explanation types. Please rephrase or ask a tutor.")
                else:
                    with st.spinner(f"🔁 Re-explaining using {next_type.replace('_', ' ').title()}..."):
                        retry_result = st.session_state.chat_chain.invoke({
                            "question": user,
                            "chat_history": st.session_state.chat_history[-5:],
                            "understood_topics": st.session_state.understood_topics,
                            "explanation_type": next_type
                        })
                    retry_answer = retry_result.get("answer", "")
                    tried.append(next_type)
                    st.session_state.explanation_attempts[topic] = tried
                    st.session_state.chat_history[i] = (user, retry_answer.strip())
                    del st.session_state[f"{key_prefix}_understood"]
                    st.session_state.last_explanation_type = next_type
                    st.rerun()

            elif st.session_state[f"{key_prefix}_understood"] is True and followups:
                st.markdown("**💡 Follow-up questions:**")
                for fq in followups:
                    if st.button(fq, key=f"{key_prefix}_fq_{fq}"):
                        st.session_state.pending_followup = fq
                        st.rerun()

with st.sidebar.expander("✅ Topics Understood", expanded=False):
    if st.session_state.understood_topics:
        for topic in st.session_state.understood_topics:
            st.checkbox(topic, value=True, disabled=True)
    else:
        st.markdown("_No topics marked as understood yet._")

with st.sidebar.expander("🎯 Helpful Explanation Types"):
    if st.session_state.explanation_success:
        for topic, ex_type in st.session_state.explanation_success.items():
            st.markdown(f"- **{topic}** → ✅ *{ex_type.replace('_', ' ').title()}*")
    else:
        st.markdown("_No successful explanation types recorded yet._")

with st.sidebar.expander("🧠 Your Preferred Explanation Types"):
    if st.session_state.successful_explanation_types:
        for t in st.session_state.successful_explanation_types:
            st.markdown(f"- {t.replace('_', ' ').title()}")
    else:
        st.markdown("_No preferred styles identified yet._")
