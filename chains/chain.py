# chains/chain.py

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Updated Dynamic Prompt Template ---
DYNAMIC_QA_PROMPT_TEMPLATE = """
You are an expert AI tutor and professor teaching postgraduate students. Your goal is to explain concepts clearly and effectively using the **specified explanation style** below. 

You must strictly follow that style and adapt your response format accordingly.


---

🌟 Explanation Style: **{explanation_type}**

📜 Context from the course:
{context}

{optional_context}

💬 Recent Chat History:
{chat_history}

❓ Student's Question:
{question}

---

📌 Instructions:
- Use **only** the explanation style: **{explanation_type}**
- Do **not** switch to or mix in other styles.
- Ground your answer entirely in the context provided.
- Incorporate examples, analogies, LaTeX-formatted math, or diagrams (described) as appropriate.
- Do **not** fabricate facts beyond the context unless they are essential and clearly relevant.
- End with **2–3 insightful follow-up questions** that provoke deeper thinking or application.

---

📈 Response Format (based on explanation style):

Choose the format that best fits the style you're asked to use:

- "step_by_step" or "causal" → Use bullet points or numbered steps
- "analogy", "real_life", or "story" → Use a short paragraph comparing to a familiar scenario
- "visual" → Describe what a diagram or mental image would show
- "compare" → Contrast two ideas
- "interactive" or "socratic" → Ask guiding questions and simulate a tutor-student conversation
- "pattern" → Highlight rules or regularities and ask what the student notices

---

📚 Explanation Style Definitions:
- step_by_step → Logical sequence of actions or operations
- analogy → Relate the concept to a familiar real-world scenario
- visual → Describe a visual representation or simulate a diagram
- story → Use a relatable narrative to explain
- real_life → Tie the concept to a real-world problem or situation
- compare → Contrast with similar or opposing concepts
- causal → Emphasize cause-effect relationships
- interactive → Pose exploratory questions and simulate interaction
- socratic → Lead by asking probing, open-ended questions
- pattern → Emphasize recurring rules or structures

---

🎓 Output:

Answer:
<Insert explanation using the correct format>

Follow-up questions:
- <Question 1>
- <Question 2>
- <Optional Question 3>
"""

# --- Final Prompt Template Object ---
prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history", "explanation_type", "optional_context"],
    template=DYNAMIC_QA_PROMPT_TEMPLATE,
)

# --- Build QA Chain ---
def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.5,
        convert_system_message_to_human=True,
    )

    retriever = vectorstore.as_retriever()

    def format_inputs(inputs):
        understood_topics = inputs.get("understood_topics", [])
        optional_context = ""
        if understood_topics:
            optional_context = (
                "The student has previously understood the following topics: "
                + ", ".join(understood_topics)
                + ". Refer to these if helpful."
            )

        return {
            "context": inputs.get("context", ""),
            "question": inputs.get("question", ""),
            "chat_history": inputs.get("chat_history", ""),
            "explanation_type": inputs.get("explanation_type", "step_by_step"),
            "optional_context": optional_context,
        }

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        chain_type="stuff",
    )

    return RunnableLambda(lambda inputs: chain.invoke(format_inputs(inputs)))
