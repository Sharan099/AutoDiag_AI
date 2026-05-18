import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from tqdm import tqdm
import os 

PROC_DIR = "data/processed"
VECTOR_DIR = "data/vector_store"

os.makedirs(VECTOR_DIR, exist_ok=True)


def load_tsb_documents():
    df = pd.read_csv(f"{PROC_DIR}/tsbs_german.csv")
    documents = []
    for _, row in df.iterrows():
        text = str(row.get("document_text", "")).strip()
        if len(text) < 80:
            continue
        doc = Document(
            page_content=text,
            metadata={
                "source":    "NHTSA_TSB",
                "make":      str(row.get("Make", "")),
                "model":     str(row.get("Model", "")),
                "year":      str(row.get("Model Year", "")),
                "component": str(row.get("NHTSA Components", "")),
                "nhtsa_id":  str(row.get("NHTSA ID Number", "")),
                "doc_type":  "tsb"
            }
        )
        documents.append(doc)
    print(f"Loaded {len(documents)} TSB documents")
    return documents


def load_recall_documents():
    df = pd.read_csv(f"{PROC_DIR}/recalls_german.csv")
    documents = []
    for _, row in df.iterrows():
        text = str(row.get("document_text", "")).strip()
        if len(text) < 80:
            continue
        doc = Document(
            page_content=text,
            metadata={
                "source":   "NHTSA_RECALL",
                "make":     str(row.get("MAKETXT", "")),
                "model":    str(row.get("MODELTXT", "")),
                "year":     str(row.get("YEARTXT", "")),
                "doc_type": "recall"
            }
        )
        documents.append(doc)
    print(f"Loaded {len(documents)} Recall documents")
    return documents





def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} docs → {len(chunks)} chunks")
    return chunks


def build_and_save_vectorstore(chunks):
    embeddings =  HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)


    BATCH = 500
    vectorstore = None

    for i in tqdm(range(0, len(chunks), BATCH), desc="Batches"):
        batch = chunks[i : i + BATCH]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    vectorstore.save_local(VECTOR_DIR)
    print(f"Saved FAISS index → {VECTOR_DIR}/")
    print(f"Total vectors stored: {vectorstore.index.ntotal}")
    return vectorstore



if __name__ == "__main__":
    tsb_docs    = load_tsb_documents()
    recall_docs = load_recall_documents()
    all_docs    = tsb_docs + recall_docs

    chunks = chunk_documents(all_docs)

    vectorstore = build_and_save_vectorstore(chunks)
    print("Vector store built. Run src/rag.py to query it.")