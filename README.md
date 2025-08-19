# RAG-Based Chatbot for Document QA (State of the Union 2024)

This repository contains a minimal Retrieval-Augmented Generation (RAG) chatbot built with LangChain and ChromaDB.
It answers questions grounded in the included text data source: the March 7, 2024 State of the Union address transcript.

## Features
- Load, split, and embed a local text file
- Store embeddings in a local Chroma vector database
- Retrieve relevant chunks and generate answers with an LLM
- Simple Gradio interface for chatting

## Getting Started

### 1) Clone and install
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Set your API key
Set your OpenAI API key as an environment variable before running:
```bash
export OPENAI_API_KEY=sk-...            # macOS/Linux
setx OPENAI_API_KEY "sk-..."            # Windows (new shell after this)
```

### 3) Run
```bash
python app.py
```

Gradio will launch a local UI in your browser.

## Project Structure
```
rag-chatbot/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ data/
   └─ 2024_state_of_the_union.txt
```

## Notes
- The vector store persists only in memory for this simple example. For production, configure persistent storage or a hosted vector DB.
- The prompt is intentionally conservative: the model responds with "I don't know" when context is insufficient.

## License
MIT
