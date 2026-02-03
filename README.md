# 🧠 Digital Twin: Agentic RAG Personal Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-Framework-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![OpenAI](https://img.shields.io/badge/LLM-GPT--4o--Mini-orange)

A production-grade **Agentic RAG (Retrieval-Augmented Generation)** system that serves as a "Digital Twin." It autonomously decides whether to answer queries based on my personal local data (Resume, Projects, Codebase) or by searching the live internet.

## 🎥 Visual Demo

> **"Mind decides. Body acts."**
> Watch how the Agentic Brain processes a user query in real-time vs. retrieving static memory.

*[watch the full architectural breakdown on LinkedIn](YOUR_LINKEDIN_POST_URL_HERE).*

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

```text
├── data/                   # Place your PDFs, CSVs, and Code files here
├── app.py                  # Main Streamlit application
├── agent_logic.py          # LangChain Agent and Tool definitions
├── vector_store.py         # ChromaDB setup and data ingestion logic
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```
---

## 🚧 Challenges & Solutions

### 1. The "Ghost Data" Problem
**Issue:** Old files remained in the Vector DB even after being deleted from the source folder, leading to outdated answers or conflicting information.

**Solution:** Implemented a robust "Nuclear Clean-up" protocol using `shutil` that wipes and rebuilds the `chroma_db` directory on initialization, ensuring 100% data synchronicity.

### 2. The "Crowding Out" Effect
**Issue:** When retrieving too many document chunks (high `k` value), irrelevant text would "crowd out" the specific answer in the context window, causing the model to miss key details ("Lost in the Middle" phenomenon).

**Solution:** Optimized the retrieval parameters (tuned `k` and `score_threshold`) and refined chunk sizes to ensure only high-quality, relevant context reaches the LLM.

### 3. Hallucinations on Code
**Issue:** The model initially ignored local `.py` files and gave generic coding advice (e.g., standard library usage) instead of explaining the specific custom logic in the project.

**Solution:** A strict System Prompt was engineered: *"PRIORITY: Check Context First. Do not answer from general knowledge if the answer is in the retrieved documents."*

### 4. Complex File Parsing (Image-based PDFs)
**Issue:** Standard PDF loaders returned empty strings for scanned documents or files with complex layouts, creating knowledge gaps.

**Solution:** Integrated OCR-capable preprocessing and built a custom file router that detects file types and switches between standard text extraction and OCR-based loading when necessary.
