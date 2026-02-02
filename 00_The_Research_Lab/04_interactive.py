import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

# 1. SETUP (Paste your key)
os.environ["OPENAI_API_KEY"] = (
    "OPENAI_API_KEY"
)

# 2. MODEL & PROMPT
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who remembers past conversations."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm


# 3. PERSISTENT MEMORY FUNCTION
def get_session_history(session_id: str):
    # This saves to a file named 'memory1_dhamu.json'
    return FileChatMessageHistory(f"./memory1_{session_id}.json")


conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- INTERACTIVE LOOP ---
config = {"configurable": {"session_id": "dhamu"}}

print("--- Chat Started (Type 'exit' to quit) ---")

while True:
    # This allows YOU to type in the terminal
    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit", "q"]:
        print("--- Saving Memory & Exiting ---")
        break

    # Send to AI
    response = conversation.invoke({"input": user_input}, config=config)
    print(f"AI: {response.content}")
