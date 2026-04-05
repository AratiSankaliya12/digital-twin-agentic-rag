import os
import config  # Loads environment variables from .env

# NEW IMPORT: DirectoryLoader loads everything in a folder
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. LOAD MULTIPLE PDFs
# Instead of pointing to "sample.pdf", we point to the FOLDER "assets/"
print("--- Loading All PDFs from Folder ---")

# Robust path handling (recommended)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
pdf_folder = os.path.join(ROOT_DIR, "assets")

loader = PyPDFDirectoryLoader(pdf_folder)
docs = loader.load()

print(f"Loaded {len(docs)} pages from multiple files.")

# 2. SPLIT (Same as before)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(f"Split into {len(splits)} chunks.")

# 3. VECTOR STORE (Same as before - everything gets mixed)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), persist_directory="chroma_db_multi"
)

# 4. RETRIEVER (Crucial Change for Comparison)
# search_kwargs={"k": 6} tells it to fetch 6 chunks instead of the default 4.
# This helps ensure we get info on BOTH people.
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 5. CHAIN (Same as before)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# System prompt that encourages synthesis
system_prompt = (
    "You are a helpful assistant. "
    "Use the provided context to answer the question. "
    "If the user asks to compare or find commonalities, look strictly "
    "at the retrieved context for both subjects."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# --- TEST IT ---
user_question = "What are the common skills between Arati and Vandan?"

print(f"\nQuestion: {user_question}")

response = rag_chain.invoke({"input": user_question})

print(f"Answer: {response['answer']}")
