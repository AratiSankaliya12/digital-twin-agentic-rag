# memory_openai.py
import os

# 1. New Imports for LangChain v1.0+
# We use 'langchain_classic' because ConversationChain was moved there
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryMemory

# 2. Setup your API Key
# Replace 'sk-...' with your actual key from platform.openai.com
os.environ["OPENAI_API_KEY"] = (
    "OPENAI_API_KEY"

# 3. Initialize the Brain (OpenAI)
# model="gpt-4o" is the smartest, "gpt-3.5-turbo" is cheaper
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Initialize the Memory
# This uses OpenAI to summarize the conversation as you go
summary_memory = ConversationSummaryMemory(llm=llm)

# 5. Create the Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True,  # Keeps showing you the internal 'thinking'
)

# --- START TALKING ---
print("--- Starting Conversation with OpenAI ---")

# Turn 1
user_input = "Hi, I am Dhamu. I am a student learning AI."
print(f"\nUser: {user_input}")
conversation.predict(input=user_input)

# Turn 2
user_input = "I prefer coding in Python because I find it easy."
print(f"\nUser: {user_input}")
conversation.predict(input=user_input)

# Turn 3 (Asking something that requires memory)
user_input = "Can you suggest a small project for me based on my preference?"
print(f"\nUser: {user_input}")
response = conversation.predict(input=user_input)
print(f"AI: {response}")

# --- REVEAL MEMORY ---
print("\n--- WHAT THE AI REMEMBERS (Summary) ---")
print(conversation.memory.buffer)
