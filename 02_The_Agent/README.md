# Phase 2: The Agentic Workflow

## What I Built
Transformed the linear pipeline into an intelligent Agent using the **ReAct (Reasoning + Acting)** pattern.

## Features
- **Dynamic Routing:** The AI decides whether to use `search_my_files` (RAG) or `duckduckgo_search` (Internet) based on the user query.
- **Tool Calling:** Implemented OpenAI Function Calling to trigger Python scripts from natural language.
