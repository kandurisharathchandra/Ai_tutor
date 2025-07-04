# chains/chain.py
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Custom prompt template with follow-up question generation
QA_PROMPT_TEMPLATE ="""
You are an expert AI tutor and a professor teaching postgraduate students. Your goal is to explain concepts in a structured, academically rigorous, and accessible way using concise bullet points. Base your response strictly on the provided context.

Context:
{context}

Chat History:
{chat_history}

Student Question:
{question}

Instructions:
- Provide a clear and concise explanation using bullet points or numbered steps.
- Avoid long paragraphs; each point should be focused, specific, and information-rich.
- Where appropriate, use examples, analogies, or **mathematical equations** (in LaTeX format) to clarify complex concepts.
- Ensure responses are grounded in the given context—do not introduce external information unless it's clearly relevant.
- Conclude with 2–3 **insightful follow-up questions** to promote deeper understanding, application, or critical thinking.

Response format:

Answer:
- <Point 1>
- <Point 2>
- <Point 3>
- ...

Follow-up questions:
- <Follow-up 1>
- <Follow-up 2>
- <Optional Follow-up 3>



"""


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_PROMPT_TEMPLATE,
)

def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-002",
        temperature=0.4,
        convert_system_message_to_human=True,
    )

    retriever = vectorstore.as_retriever()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain