import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. SETUP API KEY
os.environ["OPENAI_API_KEY"] = (
    "OPENAI_API_KEY"
)

# 2. LOAD THE PDF
# We assume the file is in a folder named 'data' one level up
pdf_path = "../data/Arati_Resume.pdf"
print("--- Loading PDF ---")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

# 3. SPLIT TEXT (Chunking)
# We break the PDF into chunks of 1000 characters with 200 overlap.
# Overlap ensures we don't cut a sentence in half effectively.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks.")

# 4. CREATE VECTOR STORE (The "Brain" of RAG)
# This converts text -> numbers -> database
print("--- Creating Vector Store (This may take a moment) ---")
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# Make the vector store a "Retriever" (Search Engine)
retriever = vectorstore.as_retriever()

# 5. CREATE THE RAG CHAIN
# The LLM that will write the final answer
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The System Prompt (Instructions)
system_prompt = (
    "You are a helpful assistant. "
    "Use the provided context to answer the user's question. "
    "If the question has multiple parts and the context only contains "
    "information for some of them, answer what you can and explicitly "
    "state that you do not have information about the other parts. "
    "Do not attempt to make up answers for the missing parts."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# "Stuff" chain puts the found documents into the prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- ASK A QUESTION ---
print("\n--- RAG Ready! Asking Question ---")

# CHANGE THIS QUESTION to something that is INSIDE your PDF
user_question = "tell me about Arati and Dhamu?"

response = rag_chain.invoke({"input": user_question})

print(f"Question: {user_question}")
print(f"Answer: {response['answer']}")
