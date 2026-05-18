"""
build_vectorstore.py
---------------------
Loads cleaned CSV files, converts rows to LangChain Documents,
splits them into chunks, embeds with nomic-embed-text via Ollama,
and saves the FAISS vector store to disk.

Run once. Takes 5-15 minutes for ~15 000 chunks.
For sample data (10 records) it takes ~30 seconds.

Usage: python src/build_vectorstore.py
Inputs:  data/processed/tsbs_german.csv
         data/processed/recalls_german.csv
Outputs: data/vector_store/index.faiss
         data/vector_store/index.pkl
"""

import os
import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

PROC_DIR   = os.getenv("PROCESSED_DATA_PATH", "data/processed")
VECTOR_DIR = os.getenv("VECTOR_STORE_PATH",   "data/vector_store")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")

os.makedirs(VECTOR_DIR, exist_ok=True)


def load_tsb_documents():
    path = f"{PROC_DIR}/tsbs_german.csv"
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        text = str(row.get("document_text", "")).strip()
        if len(text) < 80:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source":    "NHTSA_TSB",
                "make":      str(row.get("Make", "")),
                "model":     str(row.get("Model", "")),
                "year":      str(row.get("Model Year", "")),
                "component": str(row.get("NHTSA Components", "")),
                "nhtsa_id":  str(row.get("NHTSA ID Number", "")),
                "doc_type":  "tsb",
            },
        ))
    print(f"  Loaded {len(docs)} TSB documents")
    return docs


def load_recall_documents():
    path = f"{PROC_DIR}/recalls_german.csv"
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        text = str(row.get("document_text", "")).strip()
        if len(text) < 80:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source":   "NHTSA_RECALL",
                "make":     str(row.get("MAKETXT", "")),
                "model":    str(row.get("MODELTXT", "")),
                "year":     str(row.get("YEARTXT",  "")),
                "doc_type": "recall",
            },
        ))
    print(f"  Loaded {len(docs)} Recall documents")
    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split {len(documents)} docs -> {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    print(f"  Connecting to Ollama at {OLLAMA_URL} ...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    print("  Embedding in batches of 200 ...")
    BATCH = 200
    vectorstore = None

    for i in tqdm(range(0, len(chunks), BATCH), desc="  Embedding"):
        batch = chunks[i : i + BATCH]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    vectorstore.save_local(VECTOR_DIR)
    print(f"  Saved FAISS index -> {VECTOR_DIR}/")
    print(f"  Total vectors: {vectorstore.index.ntotal}")
    return vectorstore


if __name__ == "__main__":
    print("Step 1: Loading documents ...")
    tsb_docs    = load_tsb_documents()
    recall_docs = load_recall_documents()
    all_docs    = tsb_docs + recall_docs

    print("\nStep 2: Chunking ...")
    chunks = chunk_documents(all_docs)

    print("\nStep 3: Embedding and saving to FAISS ...")
    build_vectorstore(chunks)

    print("\nDone. Run: python src/rag.py")
