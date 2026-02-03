# Phase 1: The RAG Pipeline

## What I Built
A retrieval-based chatbot using LangChain and ChromaDB. It ingests local PDFs (Resumes) and answers questions based strictly on their content.

## Key Challenges Solved
- **"Ghost Data":** The vector database kept retaining deleted files. I implemented a `shutil` cleanup script to flush the DB on every run to ensure data consistency.
- **Hallucinations:** The model tried to use general Python knowledge instead of my specific code files. I engineered a strict system prompt to force context-priority.
