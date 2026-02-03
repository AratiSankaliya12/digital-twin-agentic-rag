import os
import sys
import shutil  # <--- NEW: Tool to delete folders

# 1. IMPORTS
# Loaders & Splitters
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Chains & Prompts
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Memory & Persistence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

# --- CONFIGURATION ---
os.environ["OPENAI_API_KEY"] = (
    "OPENAI_API_KEY"
)

DATA_FOLDER = "../data/"
PERSIST_DIRECTORY = "./chroma_db"  # Where to save the vector database on disk


# --- PART 1: THE KNOWLEDGE BASE ---
# --- NEW IMPORTS FOR DIFFERENT FILE TYPES ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredImageLoader,
)


def setup_vectorstore():
    # 1. Clean up old database
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"--- 0. Flushing old memory: Deleting '{PERSIST_DIRECTORY}' ---")
        shutil.rmtree(PERSIST_DIRECTORY)

    print(f"--- 1. Scanning '{DATA_FOLDER}' for ALL supported files ---")

    all_docs = []

    # 2. Walk through the directory file by file
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            loader = None

            try:
                # --- THE ROUTER (Decides which tool to use) ---
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)

                elif file_ext == ".docx":
                    loader = Docx2txtLoader(file_path)

                elif file_ext == ".csv":
                    loader = CSVLoader(file_path)

                elif file_ext in [
                    ".txt",
                    ".py",
                    ".sh",
                    ".md",
                    ".json",
                    ".log",
                    ".java",
                    ".c",
                ]:
                    # Code files are just text files!
                    loader = TextLoader(file_path)

                elif file_ext in [".jpg", ".png", ".jpeg"]:
                    # REQUIRES TESSERACT INSTALLED ON UBUNTU
                    # sudo apt-get install tesseract-ocr
                    # loader = UnstructuredImageLoader(file_path)
                    print(f" SKIPPING IMAGE: {file} (OCR Engine not installed)")
                    continue

                else:
                    print(f" SKIPPING UNKNOWN FILE: {file}")
                    continue

                # --- LOAD THE FILE ---
                print(f"   > Loading: {file} ({file_ext})")
                file_docs = loader.load()
                all_docs.extend(file_docs)

            except Exception as e:
                print(f"   ❌ ERROR loading {file}: {e}")

    # 3. Validation
    if not all_docs:
        print("ERROR: No valid documents found!")
        sys.exit()

    print(f"   > Total Pages/Files Loaded: {len(all_docs)}")

    # 4. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    print(f"   > Split into {len(splits)} chunks.")

    # 5. Store
    print("--- 2. Building Fresh Vector Database ---")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )
    return vectorstore


# --- PART 2: THE BRAINS (Chains) ---
def create_conversational_rag_chain(vectorstore):
    """
    Creates the dual-brain chain:
    Brain 1: Rewrites the question based on history.
    Brain 2: Answers the question based on docs.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1. The Retriever (Search Engine)
    # k=6 allows it to find multiple docs (good for comparing Arati vs Others)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 2. Brain 1: The "History Aware" Reformulator
    # This prompt teaches the LLM how to rewrite questions.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),  # Takes the chat history
            ("human", "{input}"),
        ]
    )

    # This chain does the rewriting
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. Brain 2: The "Answerer"
    # This prompt tells the LLM how to answer using the found docs.
    # Inside create_conversational_rag_chain function
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, say that you don't know. "
        "DO NOT use your own outside knowledge or generate generic code examples "
        "unless they are explicitly present in the context."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),  # Context is needed here too
            ("human", "{input}"),
        ]
    )

    # This chain generates the final answer
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 4. Connect everything
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# --- PART 3: MEMORY PERSISTENCE ---
def get_session_history(session_id: str):
    # Saves chat history to a JSON file
    return FileChatMessageHistory(f"./memory_{session_id}.json")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Data
    vectorstore = setup_vectorstore()

    # 2. Setup Logic
    rag_chain = create_conversational_rag_chain(vectorstore)

    # 3. Wrap with Memory
    final_bot = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    print("\n-------------------------------------------------")
    print("--- 🤖 ASSISTANT READY (Type 'exit' to quit) ---")
    print("-------------------------------------------------")

    # 4. Chat Loop
    # We use a session ID so the memory file is unique to 'dhamu'
    config = {"configurable": {"session_id": "dhamu"}}

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("--- Saving memory and exiting. Bye! ---")
            break

        # Invoke the bot
        # We don't need to manually handle history, the wrapper does it.
        response = final_bot.invoke({"input": user_input}, config=config)

        print(f"AI: {response['answer']}")
