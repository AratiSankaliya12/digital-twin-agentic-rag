# Phase 0: The Research Lab 🧪

Before building the production agent, I conducted isolated experiments to master the two core pillars of Generative AI: **Context (Memory)** and **Retrieval (RAG)**.

## Part 1: The 4-Type Memory Journey
I built four distinct prototypes to understand how LLMs retain information:

1.  **Interactive Memory (`01_interactive.py`):**
    - *Concept:* Ephemeral RAM.
    - *Learning:* Built a loop that remembers the user's name within a single script run. Data is lost on exit.
2.  **Persistent Memory (`02_persistent.py`):**
    - *Concept:* Long-term Storage.
    - *Learning:* Implemented `FileChatMessageHistory` to save chat logs to a JSON file, allowing the bot to pick up conversations days later.
3.  **Legacy OpenAI Memory (`03_legacy_openai.py`):**
    - *Concept:* The "Old Way".
    - *Learning:* Explored the deprecated LangChain functions to understand the history of the framework.
4.  **Modern OpenAI Memory (`04_modern_openai.py`):**
    - *Concept:* The "New Way".
    - *Learning:* Mastered `RunnableWithMessageHistory` and the new Chain syntax which is more efficient and scalable.

## Part 2: The RAG Evolution
I evolved my retrieval strategy through three stages:

1.  **Basic RAG (`01_basic_rag.py`):**
    - Simple "Load PDF -> Split -> Embed -> Retrieve" pipeline.
    - *Limitation:* Could only handle one file type.
2.  **Multi-Doc Router (`02_multidoc_router.py`):**
    - Built a custom router to ingest `.pdf`, `.csv`, `.txt`, and `.py` files simultaneously.
    - *Optimization:* Solved the "Image PDF" issue where scanned resumes returned empty text.
3.  **Agentic RAG (See Phase 2):**
    - The final evolution where the RAG pipeline became a *tool* that an Agent calls dynamically.
