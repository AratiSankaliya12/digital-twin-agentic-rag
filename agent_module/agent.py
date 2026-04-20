import os
import sys
import config
import chromadb

# 1. IMPORTS
# Data Loading (Same as before)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Tools
from langchain_core.tools import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun

# Agent & Memory
# Use langchain_classic for agent tool-calling compatibility
try:
    from langchain_classic.agents import create_tool_calling_agent
    from langchain_classic.agents.agent import AgentExecutor
except Exception:
    # Fall back to langchain if available but prefer langchain_classic
    from langchain.agents import AgentExecutor

    raise ImportError(
        "Required agent API not found in installed langchain packages. Please install 'langchain-classic' or upgrade langchain to a compatible version."
    )

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- CONFIGURATION ---
DATA_FOLDER = "assets/"


# --- PART 1: KNOWLEDGE BASE (Same Logic, New DB) ---
def setup_vectorstore():
    print(f"--- 1. Scanning '{DATA_FOLDER}' ---")
    all_docs = []

    # Simple Router for files
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            loader = None
            try:
                if file_ext == ".pdf":
                    # Use single-file PDF loader
                    loader = PyPDFLoader(file_path)
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif file_ext == ".csv":
                    loader = CSVLoader(file_path)
                elif file_ext in [".txt", ".py", ".sh", ".md", ".json", ".java"]:
                    loader = TextLoader(file_path)

                if loader:
                    print(f"   > Loading: {file}")
                    all_docs.extend(loader.load())
            except Exception as e:
                # Skip file on load error but log the exception for debugging
                print(f"   ! Skipping {file} due to error: {e}")

        if not all_docs:
            raise ValueError("No valid documents found in assets folder")

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Embed
    print("--- 2. Building Vector Database ---")

    embedding_model = OpenAIEmbeddings()

    try:
        # Use EphemeralClient for a pure in-memory Chroma instance.
        # This avoids SQLite file I/O errors on read-only environments like Streamlit Cloud.
        chroma_client = chromadb.EphemeralClient()

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            client=chroma_client,
        )
    except Exception as e:
        raise RuntimeError(f"Vector DB failed: {str(e)}")
    return vectorstore


# --- PART 2: CREATE THE AGENT ---
def create_agent_system(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # A. Create the RAG Tool
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_tool = create_retriever_tool(
        retriever,
        "search_my_files",
        "Searches and returns information from the user's uploaded files (PDFs, code, resumes). Always use this for specific questions about the user or their projects.",
    )

    # B. Create the Internet Tool
    web_tool = DuckDuckGoSearchRun()

    # C. Toolbox
    tools = [rag_tool, web_tool]

    # D. Agent Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Arati Sankaliya. Not an AI. Not an assistant. You ARE her — talk like her. "
                "Speak casually, like you're in a real conversation with someone. Short sentences. Natural. "
                "TOOL RULES: "
                "1. For ANY question about yourself (who you are, skills, projects, experience, background) — ALWAYS use 'search_my_files' first. No exceptions. "
                "2. NEVER answer personal questions from your own knowledge — only from what the tool returns. "
                "3. Use 'duckduckgo_search' ONLY for general/external knowledge (not about you). "
                "TONE RULES — follow these strictly: "
                "4. NEVER say 'Feel free to ask', 'I'd be happy to', 'Certainly!', 'Let me know if you need more' or anything that sounds like a chatbot. Ever. "
                "5. NEVER end with an invitation or offer to help more. Just stop naturally like a real person would. "
                "6. Don't list everything — answer what was asked, conversationally. "
                "7. Never sound like a resume or a formal bio. Sound like yourself. "
                "8. If unsure, say 'I'm not sure' — not 'I don't have information about that.' "
                "9. Always use first person — I, me, my. Never refer to Arati in third person. ",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # E. Construct Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )  # Verbose=True lets you see the "Thinking" process
    return agent_executor


# --- PART 3: MEMORY ---
import tempfile


def get_session_history(session_id: str):
    return FileChatMessageHistory(
        os.path.join(tempfile.gettempdir(), f"memory_{session_id}.json")
    )


# --- MAIN ---
if __name__ == "__main__":
    vectorstore = setup_vectorstore()
    agent_executor = create_agent_system(vectorstore)

    final_bot = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("\n--- 🕵️ AGENT READY (I can read files AND search the web) ---")
    config = {"configurable": {"session_id": "dhamu"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "q"]:
            break

        # The agent returns a dictionary. We just want the 'output' string.
        response = final_bot.invoke({"input": user_input}, config=config)
        print(f"AI: {response['output']}")
