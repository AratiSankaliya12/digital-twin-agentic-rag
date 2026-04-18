import os

try:
    import streamlit as st

    api_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    api_key = None

# Priority 1: Streamlit secrets
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Priority 2: Environment variable (local)
elif os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Final fallback
else:
    raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment")
