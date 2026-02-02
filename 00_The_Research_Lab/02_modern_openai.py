import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 1. SETUP API KEY
os.environ["OPENAI_API_KEY"] = (
    "OPENAI_API_KEY"

# 2. SETUP THE MODEL
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. SETUP THE MEMORY STORAGE
# We need a place to store the chats. We use a simple dictionary.
# Key = Session ID (User ID), Value = History of that user
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 4. CREATE THE PROMPT (Explicitly)
# This is the "Blueprint".
# {history} is where the past messages go.
# {input} is the new user question.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant named Dhamu's Helper."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# 5. CREATE THE CHAIN (The "Pipe")
# Logic: Prompt -> Model
chain = prompt | llm

# 6. WRAP WITH MEMORY
# We wrap the simple chain with the history manager.
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- START TALKING ---
print("--- Starting Conversation (LCEL Modern Way) ---")

# We must provide a "session_id". This allows the bot to remember distinct users.
config = {"configurable": {"session_id": "dhamu_session_1"}}

# Turn 1
response1 = conversation.invoke({"input": "Hi, I am Dhamu."}, config=config)
print(f"\nUser: Hi, I am Dhamu.\nAI: {response1.content}")

# Turn 2
response2 = conversation.invoke({"input": "I like Python."}, config=config)
print(f"\nUser: I like Python.\nAI: {response2.content}")

# Turn 3 (Testing Memory)
response3 = conversation.invoke({"input": "What is my name?"}, config=config)
print(f"\nUser: What is my name?\nAI: {response3.content}")
