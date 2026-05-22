# AutoDiag AI 🔧

> Multi-agent AI system for automotive fault diagnosis — built for German OEMs

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/YOUR_USERNAME/autodiag-ai)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What it does

Mechanics enter a VIN and DTC (fault) codes. The system:

1. Decodes the VIN via NHTSA API → identifies make, model, year
2. Retrieves relevant TSBs from a FAISS vector store (RAG)
3. Runs 4 specialised CrewAI agents — diagnosis, root cause, parts, report
4. Returns a structured repair report

**Directly addresses the €4B annual warranty cost problem at German OEMs.**

## Tech stack

| Layer | Tools |
|-------|-------|
| LLM | `llama3.2:1b` via Ollama (local) |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace) |
| RAG | LangChain + FAISS |
| Agents | CrewAI (4 agents) |
| Orchestration | LangGraph |
| Observability | LangSmith |
| Evaluation | RAGAS |
| API | FastAPI |
| UI | Streamlit |
| Deploy | Docker + Kubernetes + Azure AKS |

## Quick start (local)

```bash
# 1. Pull Ollama models
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill environment variables
cp .env.example .env

# 4. Create sample data and build vector store
python src/create_sample_data.py
python src/build_vectorstore.py

# 5. Run the Streamlit app
streamlit run app.py
```

## Run with Docker

```bash
docker compose up --build
```

Opens at http://localhost:8501. Ollama runs as a sidecar container.

## Project structure

```
autodiag-ai/
├── app.py                      # Streamlit UI
├── main.py                     # FastAPI REST API
├── src/
│   ├── create_sample_data.py   # Sample automotive data
│   ├── build_vectorstore.py    # Build FAISS index
│   ├── rag.py                  # RAG retrieval chain
│   ├── agents.py               # CrewAI 4-agent crew
│   ├── graph.py                # LangGraph pipeline
│   ├── evaluate.py             # RAGAS evaluation
│   └── mcp_servers/            # MCP tool servers
├── data/
│   └── processed/              # Cleaned CSV data
└── k8s/                        # Kubernetes manifests
```

## Live demo

A demo version runs on Hugging Face Spaces using the HuggingFace Inference API
instead of local Ollama (no GPU required).

👉 **[Try the live demo](https://huggingface.co/spaces/YOUR_USERNAME/autodiag-ai)**


