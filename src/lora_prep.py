"""
lora_prep.py
------------
Prepares a JSONL instruction-tuning dataset from NHTSA complaints
for LoRA fine-tuning of llama3.2:1b on automotive domain language.

Output format (Alpaca-style instruction tuning):
  {"instruction": "...", "input": "...", "output": "..."}

The fine-tuned model becomes better at understanding DTC codes
and automotive repair terminology specific to German OEMs.

Usage: python src/lora_prep.py
Inputs:  data/processed/tsbs_german.csv (as training source)
Outputs: data/lora_training/train.jsonl
         data/lora_training/eval.jsonl
"""

import os
import json
import pandas as pd
from typing import List, Dict

PROC_DIR  = "data/processed"
LORA_DIR  = "data/lora_training"
os.makedirs(LORA_DIR, exist_ok=True)

EVAL_FRACTION = 0.1


def tsb_to_instruction(row: pd.Series) -> Dict:
    """Convert one TSB row into an instruction-tuning example."""
    make      = str(row.get("Make", ""))
    model     = str(row.get("Model", ""))
    year      = str(row.get("Model Year", ""))
    component = str(row.get("NHTSA Components", ""))
    summary   = str(row.get("document_text", ""))

    if not summary or len(summary) < 80:
        return None

    instruction = (
        f"You are an automotive diagnostic expert. "
        f"A {year} {make} {model} has a fault in the {component} system. "
        f"What does the following Technical Service Bulletin tell us, "
        f"and what is the recommended repair?"
    )
    output = summary.split("Summary:")[-1].strip() if "Summary:" in summary else summary

    return {
        "instruction": instruction,
        "input":       f"TSB data: {summary[:300]}",
        "output":      output,
    }


def create_qa_pairs() -> List[Dict]:
    """Create question-answer pairs from TSB data."""
    pairs = [
        {
            "instruction": "What does DTC code P0301 mean on a BMW?",
            "input":  "",
            "output": (
                "P0301 is a Cylinder 1 Misfire Detected fault. On BMW vehicles, "
                "this is commonly caused by a faulty crankshaft position sensor "
                "connector (part #13627797870), a failed ignition coil, or a bad "
                "spark plug. The connector should be checked first as it is "
                "susceptible to vibration damage on the B58 engine."
            ),
        },
        {
            "instruction": "Explain DTC P0420 for a BMW 5 Series.",
            "input":  "",
            "output": (
                "P0420 means Catalyst System Efficiency Below Threshold Bank 1. "
                "On BMW B48 engines, this is most often caused by a failed upstream "
                "oxygen sensor (part #11787589071). Always check for exhaust leaks "
                "before replacing the catalytic converter. Replace the O2 sensor, "
                "clear codes, and run a complete drive cycle of 100km to confirm repair."
            ),
        },
        {
            "instruction": "What causes P0087 on a Volkswagen Golf and how is it fixed?",
            "input":  "",
            "output": (
                "P0087 is Fuel Rail Pressure Too Low. On the VW Golf EA888 engine, "
                "this is caused by wear of the high-pressure fuel pump (HPFP). "
                "Replace the HPFP with part #06K127025M. After replacement, run "
                "fuel system adaptation using VCDS diagnostic software."
            ),
        },
        {
            "instruction": "What is a Technical Service Bulletin (TSB)?",
            "input":  "",
            "output": (
                "A Technical Service Bulletin (TSB) is a document issued by a vehicle "
                "manufacturer to authorised dealers describing a known fault and its "
                "official repair procedure. TSBs are not recalls — they are not mandatory "
                "but represent the manufacturer's recommended fix for a known issue. "
                "They include part numbers, repair steps, and diagnostic procedures."
            ),
        },
        {
            "instruction": "How do you perform a DPF regeneration on a VW Passat TDI?",
            "input":  "DTC P2002 is present",
            "output": (
                "P2002 indicates the diesel particulate filter (DPF) efficiency is below "
                "threshold. First, perform a forced DPF regeneration using VCDS diagnostic "
                "tool. Then drive at motorway speed (above 100 km/h) for at least 30 minutes "
                "to allow passive regeneration. If the soot mass exceeds 60g or pressure "
                "differential is over 60 mbar, the DPF must be replaced "
                "(part #5Q0254700HX)."
            ),
        },
    ]
    return pairs


def prepare_dataset():
    print("Loading TSB data ...")
    path = f"{PROC_DIR}/tsbs_german.csv"
    df   = pd.read_csv(path)

    examples = []
    for _, row in df.iterrows():
        ex = tsb_to_instruction(row)
        if ex:
            examples.append(ex)

    qa_pairs = create_qa_pairs()
    examples.extend(qa_pairs)

    print(f"Total examples: {len(examples)}")

    split_idx = int(len(examples) * (1 - EVAL_FRACTION))
    train_set = examples[:split_idx]
    eval_set  = examples[split_idx:]

    train_path = f"{LORA_DIR}/train.jsonl"
    eval_path  = f"{LORA_DIR}/eval.jsonl"

    with open(train_path, "w") as f:
        for ex in train_set:
            f.write(json.dumps(ex) + "\n")

    with open(eval_path, "w") as f:
        for ex in eval_set:
            f.write(json.dumps(ex) + "\n")

    print(f"Train: {len(train_set)} examples -> {train_path}")
    print(f"Eval:  {len(eval_set)} examples  -> {eval_path}")
    print("\nNext step: use train.jsonl with SFTTrainer + LoRA on Google Colab.")
    print("See docs/autodiag_guide.tex Section 7 for the training script.")


if __name__ == "__main__":
    prepare_dataset()
