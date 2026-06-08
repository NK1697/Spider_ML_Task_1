# Applied ML Domain – Retrieval Augmented Generation (RAG)

This project implements a Retrieval-Augmented Generation (RAG) system over multiple research papers related to NLP and Large Language Models.

## Objective

To build a system that can:
- Ingest multiple research papers (PDFs)
- Retrieve relevant context using semantic search
- Generate answers using an LLM
- Support cross-paper comparison

## System Architecture

1. PDF Loading (PyMuPDF)
2. Text Chunking
3. Embedding Generation (SentenceTransformers)
4. Vector Storage (FAISS with cosine similarity)
5. Retrieval of top-k relevant chunks
6. LLM-based response generation (Qwen via Ollama)

## Features

### Normal QA Mode
- Answers questions using retrieved context

### Comparison Mode
- Compares multiple research papers
- Highlights differences across papers
- Produces structured responses

### Hallucination Control
- Uses cosine similarity threshold
- Returns:
 "Not in given data" if irrelevant

## Example Queries

- How does self-attention differ from recurrence?
- What problem does RAG solve?
- Difference between BERT and GPT
- How does LoRA reduce training cost?

## Tech Stack

- FAISS (vector search)
- SentenceTransformers (embeddings)
- PyMuPDF (PDF parsing)
- Qwen LLM (Ollama)
- Streamlit (UI)
