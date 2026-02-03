import streamlit as st
import os

# Import the Brain (Your existing Agent code)
# We import specific functions so we don't run the terminal loop
from agent_bot import setup_vectorstore, create_agent_system
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="My AI Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Personal AI Assistant")
st.markdown("I can read your **Files** 📂 and search the **Internet** 🌍.")

# --- 1. SESSION STATE (The Browser Memory) ---
# Streamlit refreshes the script on every click.
# We must save the chat history in 'st.session_state' so it doesn't vanish.
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 2. LOAD THE BRAIN (Cached) ---
# We use @st.cache_resource so it only loads the database ONCE (start-up),
# not every time you ask a question. This makes it fast.
@st.cache_resource
def load_agent():
    vectorstore = setup_vectorstore()
    agent_executor = create_agent_system(vectorstore)
    return agent_executor


agent_executor = load_agent()

# --- 3. DISPLAY CHAT HISTORY ---
# Re-draw all previous bubbles every time the screen updates
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. HANDLE USER INPUT ---
if user_input := st.chat_input("Ask me anything about Arati or the world..."):
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    # Save to browser history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # B. Generate AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Show a "Thinking" spinner while the Agent decides tools
        with st.spinner("Thinking... (Checking Files & Internet)"):
            try:
                # Wrap with Memory (Just like in terminal)
                def get_session_history(session_id: str):
                    return FileChatMessageHistory(f"./memory_agent_{session_id}.json")

                agent_with_memory = RunnableWithMessageHistory(
                    agent_executor,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )

                # RUN THE AGENT
                config = {"configurable": {"session_id": "streamlit_user"}}
                response = agent_with_memory.invoke(
                    {"input": user_input}, config=config
                )
                full_response = response["output"]

                # Display Result
                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                message_placeholder.error(full_response)

    # Save AI Message to browser history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
