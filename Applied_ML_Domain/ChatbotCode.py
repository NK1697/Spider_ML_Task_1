#LIBRARIES

import os
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import streamlit as st

#HYPERPARAMETERS

PDF_FOLDER = "/Users/niravkarahe/Desktop/Spider ML/Files_RAG"
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b-instruct"
K = 5
SIM_THRESHOLD = 0.2

#CHUNKING with overlap to keep context

def chunk_text(text, chunk_size=400, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def load_rag_system():

    #INGESTING PDFs
    docs = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            reader = fitz.open(path)
            text = ""
            for page in reader:
                text += page.get_text("text") + "\n"
            docs.append({"source": file,"text": text})


    all_chunks = []

    #Adding citations to chunks
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, c in enumerate(chunks):
            all_chunks.append({"text": c,"source": doc["source"],"id": i})

    #EMBEDDING
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts,convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings,axis=1,keepdims=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return model, index, all_chunks

@st.cache_resource
def get_system():
    return load_rag_system()

def retrieve(query, k=K):
    #ENCODING QUERY
    qv = model.encode([query], convert_to_numpy=True)
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    scores, indices = index.search(qv, k)
    grouped = {}

    for score, idx in zip(scores[0], indices[0]):
        doc = all_chunks[idx]
        paper = doc["source"]
        if paper not in grouped:
            grouped[paper] = []
        grouped[paper].append({"text": doc["text"],"score": float(score)})

    return grouped, scores[0][0]

def build_context(grouped):
    context = ""
    for paper, chunks in grouped.items():
        context += f"\n[PAPER={paper}]\n"
        for c in chunks:
            context += c["text"] + "\n"
        context += "\n[/PAPER]\n"

    return context

def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": LLM_MODEL,"messages": [{"role": "user", "content": prompt}],"stream": False})

    return response.json()["message"]["content"]


def rag(query, grouped, compare_mode=False):
    context = build_context(grouped)
    if not compare_mode:
        prompt = f"""
You are a strict QA system for research papers.

Rules:

1. Use ONLY information inside PAPER blocks.
2. Do NOT use external knowledge.
3. If answer is not present, respond exactly:
4. Every bullet point MUST end with a citation.
5. EXPLAIN ANSWERS IN DETAIL.

Citation format:

[paper.pdf]

Context:
{context}

Question:
{query}

Output:
- Bullet points only
- At least 3 bullet points with explaination
- Every bullet must contain a citation

Example:

- Self-attention connects all positions directly. [attention_is_all_you_need.pdf]

- Recurrence requires sequential processing. [attention_is_all_you_need.pdf]
"""

    else:

        prompt = f"""
You are a research assistant.
Use ONLY the information inside PAPER blocks.

Question:
{query}

Context:
{context}

Instructions:

1. Explain each paper separately.
2. Compare them.
3. Create a summary table.
4. Every statement must include its source paper.

Example:

- BERT uses bidirectional attention. [BERT.pdf]

- GPT uses autoregressive generation. [GPT3.pdf]

Do not make any statement without a citation.
"""

    return ask_llm(prompt)

#USER INTERFACE

st.title("Multi-Paper RAG System")
query = st.text_input("Enter your question")
compare_mode = st.toggle("Comparison Mode")
model, index, all_chunks = get_system()
if st.button("Run"):
    grouped, top_score = retrieve(query, k=15 if compare_mode else 5)

    if not grouped:
        st.write("Not in given data")
        st.stop()

    if not compare_mode and top_score < SIM_THRESHOLD:
        st.write("Not in given data")
        st.stop()

    answer = rag(query,grouped, compare_mode)
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Retrieved Papers")
    for paper in grouped.keys():
        st.write(paper)