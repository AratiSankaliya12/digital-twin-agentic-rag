#!/bin/zsh
# Arati's Ubuntu Development Workflow

# 1. Project Initialization
mkdir -p "$PROJECT_NAME" && cd "$PROJECT_NAME"

# 2. Virtual Env (Crucial for Ubuntu stability)
python3 -m venv venv
source venv/bin/activate

# 3. Environment Setup
pip install --upgrade pip
echo "Environment ready for AI development."