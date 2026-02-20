# 🧠 Digital Twin: Agentic RAG Personal Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-Framework-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![OpenAI](https://img.shields.io/badge/LLM-GPT--4o--Mini-orange)

A production-grade **Agentic RAG (Retrieval-Augmented Generation)** system that serves as a "Digital Twin." It autonomously decides whether to answer queries based on my personal local data (Resume, Projects, Codebase) or by searching the live internet.

## 🎥 Visual Demo

> **"Mind decides. Body acts."**
> Watch how the Agentic Brain processes a user query in real-time vs. retrieving static memory.

*[watch the full architectural breakdown on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7424666816813223937/?originTrackingId=dKu36J6UxoOKh2MaveFe1Q%3D%3D).*

---

## 🚀 Key Features

* **🕵️‍♂️ Agentic Workflow (ReAct Pattern):**

  Unlike standard RAG chains, this system uses an Agent that reasons before acting. It dynamically selects tools based on user intent:
  * `search_my_files`: For questions about "Arati" (Resume, specific coding projects).
  * `duckduckgo_search`: For real-time queries (e.g., "Current Bitcoin price").
  * **Direct Answer**: For chit-chat or general knowledge.

* **📂 Multi-Modal "Universal Router":**

  Custom file ingestion that supports more than just PDFs. It automatically detects and routes:
  * `.pdf` (Resumes/Docs)
  * `".txt",
                    ".py",
                    ".sh",
                    ".md",
                    ".json",
                    ".log",
                    ".java",
                    ".c"` (Codebases)
  * `.csv` (Data spreadsheets)

* **🧠 Persistent Memory:**

  Implements `FileChatMessageHistory` to remember context across conversation turns (e.g., "What was the last thing I asked you?").

* **🧹 "Nuclear" Data Cleanup:**

  Automated protocol to flush and rebuild the Vector DB on restart, solving the "Ghost Data" issue where deleted files persisted in embeddings.

* **🛡️ Hallucination Control:**

  Engineered system prompts to prioritize local context over pre-trained knowledge.
---

## 🛠️ Tech Stack

* **LLM:** GPT-4o-mini (via OpenAI API)
* **Orchestration:** LangChain (Python)
* **Vector Database:** ChromaDB (Local persistence)
* **Frontend:** Streamlit
* **Tools:** DuckDuckGo Search, PyPDF, Custom File Loaders

---

## 🏗️ Architecture

The system follows a **ReAct (Reasoning + Acting)** loop:

1.  **Input:** User asks a question.
2.  **Thought:** The LLM analyzes the query to determine the required domain.
3.  **Action:**
    * If *Personal*, it queries the **ChromaDB** vector store.
    * If *External*, it queries **DuckDuckGo**.
4.  **Observation:** The tool returns raw data.
5.  **Final Answer:** The LLM synthesizes the data into a natural language response.

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/AratiSankaliya12/digital-twin-agentic-rag.git
    cd digital-twin-agentic-rag
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_sk_key_here
    ```

5.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

This repository is organized into phases, mirroring my learning journey from basic experiments to a production microservice.

```text
├── 00_The_Research_Lab/       # Phase 0: R&D
│   ├── Memory_Experiments/    # Prototypes for Interactive vs Persistent Memory
│   └── RAG_Experiments/       # Evolution from Basic PDF RAG to Multi-Doc Routers
│
├── 01_The_Pipeline/           # Phase 1: The Core
│   └── main.py                # Basic RAG bot (Fixed "Ghost Data" & Hallucinations)
│
├── 02_The_Agent/              # Phase 2: The Brain
│   └── agent.py               # ReAct Agent with Tool Calling (File Search + DuckDuckGo)
│
├── 03_The_Interface/          # Phase 3: The Face
│   └── app.py                 # Streamlit Web App with Session State & Caching
│
├── 04_The_Production_API/     # Phase 4: The Microservice
│   ├── server.py              # FastAPI Backend (REST API)
│   └── rag_core.py            # Decoupled Agent Logic
│
├── assets/                    # Screenshots & Demo Videos
└── README.md                  # Documentation
```
---

## 🚧 Challenges & Solutions

[Click here!](https://github.com/AratiSankaliya12/digital-twin-agentic-rag/tree/main/Challenges%20%26%20Solutions)
