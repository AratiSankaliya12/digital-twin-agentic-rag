import streamlit as st
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_module.agent import setup_vectorstore, create_agent_system, log_agent_steps
import importlib
import agent_module.voice_helper

importlib.reload(agent_module.voice_helper)
from agent_module.voice_helper import (
    transcribe_audio,
    generate_speech,
    clean_text_for_speech,
)
from agent_module.audio_player import render_custom_audio_player
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
st.title("Arati's Digital Twin")

st.markdown("""
Hi, I'm Arati's AI Twin

I can talk about my projects, skills, and experience,
and also explore the internet when needed

""")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hey, I'm Arati's Digital Twin. Feel free to ask me anything about my work, projects, or AI journey!",
        }
    )

# --- LOAD AGENT ---

# Clear cached agent and vectorstore
st.cache_resource.clear()


@st.cache_resource
def load_agent():
    print("Loading updated knowledge base...")

    vectorstore = setup_vectorstore()

    print("Creating agent system...")
    agent_executor = create_agent_system(vectorstore)

    print("Digital Twin is ready!")
    return agent_executor


agent_executor = load_agent()


# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.title("Voice Settings")
    enable_voice = st.checkbox("Enable Voice Responses", value=True)
    voice_persona = st.selectbox(
        "Voice Persona",
        options=["nova", "shimmer", "alloy"],
        format_func=lambda x: {
            "nova": "Nova (Female - Smooth and Professional)",
            "shimmer": "Shimmer (Female - Friendly and Vibrant)",
            "alloy": "Alloy (Neutral - Conversational)",
        }[x],
        index=0,
    )

    st.markdown("### Flow and Pitch Tuning")
    rate_val = st.slider(
        "Speech Speed (Rate)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d%%",
    )
    pitch_val = st.slider(
        "Voice Pitch", min_value=-20, max_value=20, value=0, step=1, format="%dHz"
    )

    rate_str = f"{'+' if rate_val >= 0 else ''}{rate_val}%"
    pitch_str = f"{'+' if pitch_val >= 0 else ''}{pitch_val}Hz"

    st.markdown("---")
    st.markdown(
        "Tip: Adjust sliders to make the voice faster/slower or change the pitch for a more natural flow."
    )

# --- INITIALIZE VOICE STATE ---
if "last_played_message_index" not in st.session_state:
    st.session_state.last_played_message_index = -1

# --- DISPLAY CHAT ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("audio"):
            # Autoplay only the newest assistant message once
            should_autoplay = False
            if (
                i == len(st.session_state.messages) - 1
                and i > st.session_state.last_played_message_index
            ):
                should_autoplay = True
                st.session_state.last_played_message_index = i
            player_html = render_custom_audio_player(
                message["audio"], autoplay=should_autoplay
            )
            st.html(player_html, unsafe_allow_javascript=True)

# --- GET USER INPUT ---
# Single unified input: chat_input's accept_audio=True puts a mic button
# natively inside the chat bar itself, recording 16kHz mono WAV (optimal for
# Whisper) directly into the same widget as typed text. This replaces the
# previous separate floating mic-toggle-button + st.audio_input() widget,
# which felt structurally disconnected from the main input. Submission
# returns a dict-like object with .text and .audio when accept_audio=True,
# instead of a plain string.
chat_submission = st.chat_input(
    "Ask me about my projects, skills, or anything you're curious about...",
    accept_audio=True,
    audio_sample_rate=16000,
)

user_input = None
if chat_submission:
    if chat_submission.audio is not None:
        with st.spinner("Transcribing voice..."):
            try:
                user_input = transcribe_audio(chat_submission.audio)
            except Exception as e:
                st.sidebar.warning(f"Voice transcription failed: {e}")
    elif chat_submission.text:
        user_input = chat_submission.text

# --- PROCESS USER INPUT ---
if user_input:
    # Save user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

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

            config = {"configurable": {"session_id": "streamlit_user_v2"}}

            response = agent_with_memory.invoke({"input": user_input}, config=config)

            # --- PRINT LOGS TO TERMINAL ---
            log_agent_steps(response)

            full_response = response["output"]

            # --- HUMANIZATION ---
            full_response = full_response.replace("Arati's", "my")

            # --- TTS GENERATION ---
            audio_bytes = None
            if enable_voice:
                with st.spinner("Twin is speaking..."):
                    try:
                        clean_text = clean_text_for_speech(full_response)
                        if clean_text:
                            audio_bytes = generate_speech(
                                clean_text,
                                voice=voice_persona,
                                rate=rate_str,
                                pitch=pitch_str,
                            )
                    except Exception as tts_err:
                        st.sidebar.warning(f"Voice generation failed: {tts_err}")

            # --- TYPING EFFECT OR INSTANT DISPLAY ---
            if enable_voice:
                message_placeholder.write(full_response)
            else:
                typed_text = ""
                for char in full_response:
                    typed_text += char
                    message_placeholder.write(typed_text)
                    time.sleep(0.006)

        except Exception as e:
            full_response = f"Error: {str(e)}"
            message_placeholder.error(full_response)
            audio_bytes = None

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response, "audio": audio_bytes}
        )
        st.rerun()

# --- FOOTER ---
st.markdown("---")
st.markdown("Built by Arati | Digital Twin Project")
