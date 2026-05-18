"""
data_prep.py
------------
Reads the raw NHTSA flat files (tab-separated .txt),
filters to German OEM makes, builds a document_text column,
and saves clean CSV files for the vector store pipeline.

Usage: python src/data_prep.py
Inputs:  data/raw/TSBS_RECEIVED_2020-2024.txt
         data/raw/RCL_FROM_2020_2024.txt
Outputs: data/processed/tsbs_german.csv
         data/processed/recalls_german.csv
"""

import os
import pandas as pd

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

GERMAN_MAKES = ["BMW", "VOLKSWAGEN", "MERCEDES-BENZ", "AUDI", "PORSCHE"]


def clean_tsbs():
    path = f"{RAW_DIR}/TSBS_RECEIVED_2020-2024.txt"
    print(f"Loading {path} ...")

    df = pd.read_csv(
        path,
        sep="\t",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
    )
    print(f"  Total rows loaded: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    df = df[df["Make"].isin(GERMAN_MAKES)].copy()
    print(f"  After make filter: {len(df)} rows")

    df = df[df["Summary"].notna()]
    df = df[df["Summary"].str.strip().str.len() > 80]

    df["document_text"] = (
        "Make: "      + df["Make"].fillna("") + "\n"
        + "Model: "     + df["Model"].fillna("") + "\n"
        + "Year: "      + df["Model Year"].astype(str) + "\n"
        + "Component: " + df["NHTSA Components"].fillna("UNKNOWN") + "\n"
        + "TSB ID: "    + df["NHTSA ID Number"].astype(str) + "\n"
        + "Summary: "   + df["Summary"].str.strip()
    )

    keep = [
        "Make", "Model", "Model Year", "NHTSA Components",
        "NHTSA ID Number", "Mfr Communication Date", "document_text",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(f"{PROC_DIR}/tsbs_german.csv", index=False)
    print(f"  Saved {len(df)} rows -> data/processed/tsbs_german.csv")


def clean_recalls():
    path = f"{RAW_DIR}/RCL_FROM_2020_2024.txt"
    print(f"Loading {path} ...")

    df = pd.read_csv(
        path,
        sep="\t",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
    )
    print(f"  Total rows loaded: {len(df)}")

    df = df[df["MAKETXT"].isin(GERMAN_MAKES)].copy()
    print(f"  After make filter: {len(df)} rows")

    df["document_text"] = (
        "RECALL — Make: " + df["MAKETXT"].fillna("") + "\n"
        + "Model: "       + df["MODELTXT"].fillna("") + "\n"
        + "Year: "        + df["YEARTXT"].astype(str) + "\n"
        + "Component: "   + df["COMPNAME"].fillna("") + "\n"
        + "Defect: "      + df["DEFECT_SUMMARY"].fillna("") + "\n"
        + "Consequence: " + df["CONSEQUENCE_SUMMARY"].fillna("") + "\n"
        + "Remedy: "      + df["CORRECTIVE_ACTION"].fillna("")
    )

    df.to_csv(f"{PROC_DIR}/recalls_german.csv", index=False)
    print(f"  Saved {len(df)} rows -> data/processed/recalls_german.csv")


if __name__ == "__main__":
    if not os.path.exists(f"{RAW_DIR}/TSBS_RECEIVED_2020-2024.txt"):
        print("NHTSA files not found in data/raw/.")
        print("Either download them from https://static.nhtsa.gov/odi/ffdd/tsbs/")
        print("or run: python src/create_sample_data.py  (uses built-in sample data)")
    else:
        clean_tsbs()
        clean_recalls()
        print("\nDone. Run: python src/build_vectorstore.py")
