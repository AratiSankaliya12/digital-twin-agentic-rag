import os
import sys
import shutil

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
os.environ["OPENAI_API_KEY"] = (
    "OPENAI_API_KEY"
)
DATA_FOLDER = "../data/"
PERSIST_DIRECTORY = "./chroma_db_agent"  # New DB folder for the agent


# --- PART 1: KNOWLEDGE BASE (Same Logic, New DB) ---
def setup_vectorstore():
    # We clean the DB to handle new files/Java/Images correctly
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"--- 0. Flushing old memory: Deleting '{PERSIST_DIRECTORY}' ---")
        shutil.rmtree(PERSIST_DIRECTORY)

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
        print("ERROR: No valid documents found!")
        sys.exit()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Embed
    print("--- 2. Building Vector Database ---")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )
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
                "You are a helpful assistant. You have access to the user's files and the internet. "
                "Use the 'search_my_files' tool FIRST if the question is about the user, their resume, or their projects. "
                "Use the 'duckduckgo_search' tool if the user asks for current events or general knowledge not in the files.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # E. Construct Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True
    )  # Verbose=True lets you see the "Thinking" process
    return agent_executor


# --- PART 3: MEMORY ---
def get_session_history(session_id: str):
    return FileChatMessageHistory(f"./memory_agent_{session_id}.json")


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
