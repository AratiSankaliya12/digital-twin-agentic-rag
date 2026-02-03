import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# This is the new tool: It saves history to a file
from langchain_community.chat_message_histories import FileChatMessageHistory

# 1. SETUP API KEY
os.environ["OPENAI_API_KEY"] = (
    "OPENAI_API_KEY"
)

# 2. SETUP THE MODEL
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 3. SETUP THE PERSISTENT STORAGE
# This function is called every time a user talks.
# It checks if a file exists for that user. If not, it creates one.
def get_session_history(session_id: str):
    # This creates a file named 'memory_dhamu.json' in your current folder
    file_path = f"./memory_{session_id}.json"
    return FileChatMessageHistory(file_path)


# 4. CREATE PROMPT & CHAIN
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

# 5. WRAP WITH HISTORY
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- THE DEMO ---
# We use the same Session ID ("dhamu").
# This allows the AI to find the specific "Diary" for you.
config = {"configurable": {"session_id": "dhamu"}}

print("--- Chat Session Started ---")

# Step 1: Ask the AI who you are.
# (If this is your first run, it won't know. If it's your second run, it will!)
user_input = "What do you know about me?"
print(f"\nUser: {user_input}")
response = conversation.invoke({"input": user_input}, config=config)
print(f"AI: {response.content}")

# Step 2: Tell it something new.
user_input = "My favorite color is Blue."
print(f"\nUser: {user_input}")
response = conversation.invoke({"input": user_input}, config=config)
print(f"AI: {response.content}")

print("\n--- Session Ended (Memory Saved to Disk) ---")
