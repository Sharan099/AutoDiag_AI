"""
rag.py
------
Loads the saved FAISS vector store and answers automotive
diagnostic queries using Retrieval-Augmented Generation.

The retriever fetches the 5 most relevant TSB/recall chunks.
The LLM (llama3.2:1b via Ollama) generates the answer from context.

Usage:
  python src/rag.py                  # runs a demo query
  from src.rag import load_rag_chain # import in other modules
"""

import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings      # ✅ Fix 2: correct embeddings class
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA                          # ✅ Fix 1: langchain_classic → langchain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# ✅ Fix 3: removed unused `from sentence_transformers import SentenceTransformer`

load_dotenv()

VECTOR_DIR  = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL",  "llama3.2:1b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

PROMPT_TEMPLATE = """You are an expert automotive diagnostic engineer for German vehicles.

Use the Technical Service Bulletins and Recall data below to answer the question.
If the answer is not in the context, say: "No TSB found for this fault code."

Context:
{context}

Question: {question}

Provide:
1. Root cause
2. Repair steps
3. Parts needed (with part numbers if available)
4. Source TSB ID"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)


def load_retriever():
    """Load the FAISS vector store and return a retriever."""
    embeddings = HuggingFaceBgeEmbeddings(                           # ✅ Fix 2: use HuggingFaceEmbeddings
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},             # recommended for BGE models
    )
    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )


def load_rag_chain():
    """Build and return a RetrievalQA chain ready to invoke."""
    retriever = load_retriever()
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.1)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def diagnose(vehicle_info: str, dtc_codes: list) -> dict:
    """Run a diagnostic query and return result + sources."""
    chain  = load_rag_chain()
    query  = (
        f"Vehicle: {vehicle_info}. "
        f"DTC codes: {', '.join(dtc_codes)}. "
        "What are the known TSBs and recommended fix?"
    )
    result = chain.invoke({"query": query})

    sources = [
        {
            "doc_type": d.metadata.get("doc_type", ""),
            "make":     d.metadata.get("make", ""),
            "model":    d.metadata.get("model", ""),
            "year":     d.metadata.get("year", ""),
            "nhtsa_id": d.metadata.get("nhtsa_id", "N/A"),
        }
        for d in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}


if __name__ == "__main__":
    print("Testing RAG chain ...")
    output = diagnose(
        vehicle_info="BMW 3 Series 2021",
        dtc_codes=["P0301"],
    )
    print("\n=== ANSWER ===")
    print(output["answer"])
    print("\n=== SOURCES ===")
    for s in output["sources"]:
        print(f"  [{s['doc_type'].upper()}] {s['make']} {s['model']} {s['year']} | ID: {s['nhtsa_id']}")