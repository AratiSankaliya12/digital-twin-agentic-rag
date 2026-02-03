import os
import shutil

# --- LOADERS ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --- TOOLS ---
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun

# --- AGENT (The Universal Fix) ---
from langchain.agents import initialize_agent, AgentType
from langchain_classic.schema import SystemMessage
from langchain_classic.prompts import MessagesPlaceholder

# --- CONFIGURATION ---
os.environ["OPENAI_API_KEY"] = "sk-proj-..."  # <--- PASTE YOUR KEY HERE
DATA_FOLDER = "../data/"
PERSIST_DIRECTORY = "./chroma_db_api"


# 1. SETUP DATABASE (Same as before)
def initialize_vectorstore():
    # Only clear if you want fresh data every restart
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    print("--- [CORE] Building Vector Database... ---")
    all_docs = []

    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            loader = None
            try:
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif file_ext == ".csv":
                    loader = CSVLoader(file_path)
                elif file_ext in [".txt", ".py", ".md", ".json", ".java", ".sh"]:
                    loader = TextLoader(file_path)

                if loader:
                    all_docs.extend(loader.load())
            except Exception:
                pass

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )
    return vectorstore


# 2. SETUP AGENT (The Robust Way)
def get_agent_executor():
    vectorstore = initialize_vectorstore()

    # A. The Brain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # B. The Tools
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_tool = create_retriever_tool(
        retriever,
        "search_my_files",
        "Searches Arati's personal files, resume, and projects.",
    )
    web_tool = DuckDuckGoSearchRun()
    tools = [rag_tool, web_tool]

    # C. The Persona (System Message)
    system_message = SystemMessage(
        content=(
            "You are the AI Assistant for Arati (Dhamu). "
            "Use 'search_my_files' for questions about her skills/projects. "
            "Use 'duckduckgo_search' for general world info. "
            "Be professional and concise."
        )
    )

    # D. The Memory Handling
    # This tells the Agent to expect a variable called 'chat_history'
    agent_kwargs = {
        "system_message": system_message,
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
    }

    # E. The Constructor (initialize_agent)
    # This function handles the specific "New vs Old" import logic internally
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,  # Uses the reliable Function Calling API
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=None,  # We manage memory externally in server.py, so we set this to None
    )

    return agent_executor
