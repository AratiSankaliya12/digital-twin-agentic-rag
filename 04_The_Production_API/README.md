# Phase 4: FastAPI Microservice

## What I Built
Decoupled the AI logic into a standalone REST API using FastAPI. This serves as the backend for my Portfolio "Digital Twin."

## Architecture
- **Endpoints:** `POST /chat` for handling queries via JSON.
- **Legacy Compatibility:** Handled version conflicts in `langchain` by implementing a `try/except` fallback for `langchain_classic` vs `langchain_community`.
- **Memory:** Implemented external session-based JSON memory storage per API user.
