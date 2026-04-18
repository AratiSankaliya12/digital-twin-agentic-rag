import os
import streamlit as st

# Try Streamlit secrets first
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Fallback to environment (local)
elif os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

else:
    raise ValueError("OPENAI_API_KEY not found")
