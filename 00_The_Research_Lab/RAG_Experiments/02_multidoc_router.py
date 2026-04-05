import os
import config  # loads .env

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# -------- PATH SETUP --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_DIR = os.path.join(ROOT_DIR, "assets")


# -------- LOAD ALL FILE TYPES --------
def load_documents(folder_path):
    docs = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)

            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)

            elif (
                file.endswith(".txt")
                or file.endswith(".py")
                or file.endswith(".java")
                or file.endswith(".sh")
            ):
                loader = TextLoader(file_path)

            elif file.endswith(".csv"):
                loader = CSVLoader(file_path)

            else:
                continue  # skip unsupported files

            docs.extend(loader.load())

        except Exception as e:
            print(f"Skipping {file}: {e}")

    return docs


print("--- Loading All Files ---")
docs = load_documents(DATA_DIR)
print(f"Loaded {len(docs)} documents.")


# -------- SPLIT --------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# -------- VECTOR STORE --------
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="chroma_multi_modal",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})


# -------- LLM --------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -------- PROMPT --------
system_prompt = (
    "You are a helpful assistant. "
    "Answer strictly based on provided context. "
    "If multiple documents are involved, combine information carefully."
    "\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)


rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


# -------- TEST --------
query = "Compare Arati and Vandan skills"

response = rag_chain.invoke({"input": query})

print("\nQuestion:", query)
print("Answer:", response["answer"])
