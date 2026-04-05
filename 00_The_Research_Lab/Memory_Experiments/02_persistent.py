import os
import config  # ✅ NEW

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

# 1. SETUP THE MODEL
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 2. SETUP THE PERSISTENT STORAGE
def get_session_history(session_id: str):
    file_path = f"./memory_{session_id}.json"
    return FileChatMessageHistory(file_path)


# 3. CREATE PROMPT & CHAIN
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

# 4. WRAP WITH HISTORY
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- THE DEMO ---
config = {"configurable": {"session_id": "dhamu"}}

print("--- Chat Session Started ---")

user_input = "What do you know about me?"
print(f"\nUser: {user_input}")
response = conversation.invoke({"input": user_input}, config=config)
print(f"AI: {response.content}")

user_input = "My favorite color is Blue."
print(f"\nUser: {user_input}")
response = conversation.invoke({"input": user_input}, config=config)
print(f"AI: {response.content}")

print("\n--- Session Ended (Memory Saved to Disk) ---")
