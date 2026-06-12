from Vector_store.retriever import Retriever
from groq import Groq

import os
import streamlit as st


def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if api_key:
        api_key = api_key.strip()

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment or Streamlit secrets")

    print("Loaded key:", api_key[:10])
    return Groq(api_key=api_key)


def build_context(chunks):
    context = ""
    citations = []

    for i, c in enumerate(chunks, start=1):
        context += f"[{i}] {c['content']}\n\n"
        citations.append(f"[{i}] Page {c['page']} ({c['modality']})")

    return context.strip(), citations


def answer_question(question):

    client = get_client()

    retriever = Retriever(
        "data/embeddings/vector.index",
        "data/embeddings/metadata.pkl",
        top_k=5
    )

    retrieved_chunks = retriever.retrieve(question)
    context, citations = build_context(retrieved_chunks)

    prompt = f"""
You are a factual assistant.

Rules:
- Use ONLY the provided context
- Do NOT add outside knowledge
- Cite sources like [1], [2]

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    return answer, citations


if __name__ == "__main__":
    q = "What caused the slowdown in economic growth?"
    ans, cites = answer_question(q)

    print("\n🧠 Answer:\n")
    print(ans)

    print("\n📚 Sources:")
    for c in cites:
        print("-", c)