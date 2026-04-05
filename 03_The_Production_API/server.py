from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_core import get_agent_executor
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# 1. DEFINE THE DATA FORMAT (The Menu)
# This tells the outside world: "I only accept JSON that looks like this."
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"  # Optional, defaults to "default_user"


class ChatResponse(BaseModel):
    answer: str
    sources: list = []  # Future proofing


# 2. START THE APP
app = FastAPI(title="Arati AI Portfolio API", version="1.0")

# 3. LOAD THE BRAIN (On Startup)
print("--- Starting Server & Loading Brain... ---")
agent_executor = get_agent_executor()
print("--- Brain Loaded! ---")


# 4. DEFINE THE ENDPOINT (The Order Taker)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # A. Setup Memory
        def get_session_history(session_id: str):
            return FileChatMessageHistory(f"./memory_api_{session_id}.json")

        agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # B. Run the Agent
        config = {"configurable": {"session_id": request.session_id}}
        result = agent_with_memory.invoke({"input": request.query}, config=config)

        # C. Return clean JSON
        return ChatResponse(answer=result["output"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 5. HEALTH CHECK (Just to see if it's alive)
@app.get("/health")
def health_check():
    return {"status": "active", "model": "Agentic RAG"}
