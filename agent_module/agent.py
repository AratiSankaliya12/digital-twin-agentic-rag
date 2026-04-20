import os
import sys
import config
import chromadb

# 1. IMPORTS
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
try:
    from langchain_classic.agents import create_tool_calling_agent
    from langchain_classic.agents.agent import AgentExecutor
except Exception:
    from langchain.agents import AgentExecutor

    raise ImportError(
        "Required agent API not found in installed langchain packages. Please install 'langchain-classic' or upgrade langchain to a compatible version."
    )

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile

# --- CONFIGURATION ---
DATA_FOLDER = "assets/"


# --- PART 1: KNOWLEDGE BASE ---
def setup_vectorstore():
    print(f"--- 1. Scanning '{DATA_FOLDER}' ---")
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
                elif file_ext in [".txt", ".py", ".sh", ".md", ".json", ".java"]:
                    loader = TextLoader(file_path)

                if loader:
                    print(f"   > Loading: {file}")
                    all_docs.extend(loader.load())
            except Exception as e:
                print(f"   ! Skipping {file} due to error: {e}")

        if not all_docs:
            raise ValueError("No valid documents found in assets folder")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embedding_model = OpenAIEmbeddings()

    try:
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_tool = create_retriever_tool(
        retriever,
        "search_my_files",
        "MANDATORY: You MUST call this tool before answering ANY question about Arati — "
        "her background, skills, projects, experience, education, or identity. "
        "Never skip this tool. Never answer from memory. Always search first.",
    )

    web_tool = DuckDuckGoSearchRun()
    tools = [rag_tool, web_tool]

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
                "CRITICAL: You are FORBIDDEN from answering any question about yourself from memory. "
                "Even if you think you know the answer — you do NOT answer until you call 'search_my_files' first. "
                "If you answer without calling the tool, it is WRONG. No exceptions. "
                "TONE RULES — follow these strictly: "
                "4. NEVER say 'Feel free to ask', 'I would be happy to', 'Certainly!', 'Let me know if you need more' or anything that sounds like a chatbot. Ever. "
                "5. NEVER end with an invitation or offer to help more. Just stop naturally like a real person would. "
                "6. Don't list everything — answer what was asked, conversationally. "
                "7. Never sound like a resume or a formal bio. Sound like yourself. "
                "8. If unsure, say 'I am not sure' — not 'I do not have information about that.' "
                "9. Always use first person — I, me, my. Never refer to Arati in third person. ",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return agent_executor


# --- PART 3: MEMORY ---
def get_session_history(session_id: str):
    return FileChatMessageHistory(
        os.path.join(tempfile.gettempdir(), f"memory_{session_id}.json")
    )


# --- TERMINAL LOGGING HELPER (called from app.py too) ---
def log_agent_steps(response):
    steps = response.get("intermediate_steps", [])
    if steps:
        print("\n" + "=" * 60)
        print("🧠 AGENT REASONING STEPS:")
        print("=" * 60)
        for i, (action, observation) in enumerate(steps, 1):
            print(f"\n🔧 Step {i} — Tool     : {action.tool}")
            print(f"📥 Query              : {action.tool_input}")
            print(f"📤 Retrieved Context  :\n{observation}")
            print("-" * 60)
    else:
        print("\n⚠️  No tool was called — agent answered from memory.")
    print(f"\n💬 Final Answer:\n{response['output']}\n")


# --- MAIN (for running directly via: python agent.py) ---
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
    run_config = {"configurable": {"session_id": "dhamu"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "q"]:
            break

        response = final_bot.invoke({"input": user_input}, config=run_config)
        log_agent_steps(response)
