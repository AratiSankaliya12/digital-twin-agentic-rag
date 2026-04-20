import streamlit as st
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_module.agent import setup_vectorstore, create_agent_system, log_agent_steps
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Digital Twin AI Assistant", page_icon="🧠", layout="centered"
)

# --- CUSTOM STYLING ---
st.markdown(
    """
<style>
[data-testid="stChatMessage"] {
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}
[data-testid="stChatMessage"][aria-label="assistant"] {
    background-color: #1e1e2f;
}
[data-testid="stChatMessage"][aria-label="user"] {
    background-color: #2b2b3d;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- HEADER ---
st.title("🧠 Arati's Digital Twin")

st.markdown(
    """
Hi, I'm **Arati's AI Twin** 👋  

I can talk about my **projects, skills, and experience**,  
and also explore the **internet when needed** 🌍  

"""
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hey 👋 I'm Arati's Digital Twin. Feel free to ask me anything about my work, projects, or AI journey!",
        }
    )


# --- LOAD AGENT ---
@st.cache_resource
def load_agent():
    vectorstore = setup_vectorstore()
    agent_executor = create_agent_system(vectorstore)
    return agent_executor


agent_executor = load_agent()

# --- DISPLAY CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- INPUT ---
if user_input := st.chat_input(
    "Ask me about my projects, skills, or anything you're curious about..."
):

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # --- RESPONSE ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:

            def get_session_history(session_id: str):
                return FileChatMessageHistory(f"./memory_agent_{session_id}.json")

            agent_with_memory = RunnableWithMessageHistory(
                agent_executor,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            config = {"configurable": {"session_id": "streamlit_user"}}

            response = agent_with_memory.invoke({"input": user_input}, config=config)

            # --- PRINT LOGS TO TERMINAL ---
            log_agent_steps(response)

            full_response = response["output"]

            # --- HUMANIZATION ---
            full_response = full_response.replace("Arati's", "my")

            # --- TYPING EFFECT ---
            typed_text = ""
            for char in full_response:
                typed_text += char
                message_placeholder.write(typed_text)
                time.sleep(0.006)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- FOOTER ---
st.markdown("---")
st.markdown("Built by Arati ❤️ | Digital Twin Project")
