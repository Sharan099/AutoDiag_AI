"""
Evaluates the quality of the RAG pipeline using RAGAS metrics.

Metrics:
  - faithfulness      : does the answer come from the retrieved docs?
  - answer_relevancy  : is the answer relevant to the question?
  - context_precision : are the retrieved chunks actually useful?

"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL",  "llama3.2:1b")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # HuggingFace model, not Ollama

TEST_CASES = [
    {
        "question":     "What causes DTC P0301 on a BMW and how to fix it?",
        "ground_truth": (
            "P0301 is cylinder 1 misfire. On BMW it is caused by a loose "
            "crankshaft position sensor connector. Fix: replace connector "
            "part #13627797870 and clear DTCs."
        ),
    },
    {
        "question":     "How do I fix P0420 catalyst efficiency fault on BMW?",
        "ground_truth": (
            "P0420 indicates catalyst system efficiency below threshold. "
            "On BMW, replace the upstream O2 sensor part #11787589071. "
            "Check for exhaust leaks first."
        ),
    },
    {
        "question":     "What is the fix for VW Golf P0087 fuel pressure fault?",
        "ground_truth": (
            "P0087 fuel pressure too low on VW Golf EA888. "
            "Replace the high pressure fuel pump part #06K127025M. "
            "Run fuel system adaptation with VCDS after replacement."
        ),
    },
    {
        "question":     "Mercedes C-Class transmission fault P0730 repair",
        "ground_truth": (
            "P0730 incorrect gear ratio on Mercedes 9G-Tronic. "
            "Replace solenoid kit A0002770101 and flush ATF fluid. "
            "Reset transmission adaptation with XENTRY."
        ),
    },
]


def collect_answers():
    """Run the RAG chain against all test cases and collect answers."""
    from src.rag import load_rag_chain

    chain = load_rag_chain()
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, case in enumerate(TEST_CASES):
        print(f"  Test {i+1}/{len(TEST_CASES)}: {case['question'][:55]}...")
        result = chain.invoke({"query": case["question"]})

        questions.append(case["question"])
        answers.append(result["result"])
        contexts.append([doc.page_content for doc in result["source_documents"]])
        ground_truths.append(case["ground_truth"])

    return questions, answers, contexts, ground_truths


def build_ragas_llm_and_embeddings():
    """
    Build RAGAS-compatible LLM and embedding wrappers.

    Key points:
    - RAGAS 0.2+ uses LangchainLLMWrapper / LangchainEmbeddingsWrapper
      but requires RunConfig(timeout=...) to handle slow local models.
    - BAAI/bge-small-en-v1.5 is a HuggingFace model -- use HuggingFaceEmbeddings,
      NOT OllamaEmbeddings. Ollama only knows models pulled via 'ollama pull'.
    """
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,
    )

    # BAAI/bge-small-en-v1.5 downloads ~130MB on first run then cached locally
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    return ragas_llm, ragas_emb


def run_evaluation():
    print("Step 1: Collecting RAG answers ...")
    questions, answers, contexts, ground_truths = collect_answers()

    print("\nStep 2: Building RAGAS dataset ...")
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })
    print(f"  Dataset: {len(dataset)} rows")

    print("\nStep 3: Building LLM and embedding wrappers ...")
    ragas_llm, ragas_emb = build_ragas_llm_and_embeddings()

    print("\nStep 4: Configuring RAGAS metrics ...")
    from ragas import RunConfig, evaluate
    # ragas.metrics.collections only supports OpenAI InstructorLLM.
    # For local models (Ollama), always use ragas.metrics directly.
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    faithfulness.llm            = ragas_llm
    answer_relevancy.llm        = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm       = ragas_llm

    metrics = [faithfulness, answer_relevancy, context_precision]
    print("  Metrics configured with local Ollama LLM")

    # CRITICAL fix for TimeoutError:
    # Default timeout is 60s. llama3.2:1b needs up to 5 minutes per call.
    # max_workers=1 runs sequentially so Ollama is not overwhelmed.
    run_config = RunConfig(
        timeout=600,
        max_retries=2,
        max_workers=1,
    )

    print("\nStep 5: Running evaluation ...")
    print("  Expected time: 10-25 min for 4 questions x 3 metrics with llama3.2:1b")
    print("  max_workers=1 means sequential — Ollama handles one call at a time\n")

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    print("\n=== RAGAS RESULTS ===")
    df = scores.to_pandas()
    result_dict = {}

    for col in ["faithfulness", "answer_relevancy", "context_precision"]:
        if col in df.columns:
            val = df[col].mean()
            result_dict[col] = round(float(val), 3) if val == val else 0.0
            status = "GOOD" if val >= 0.7 else "NEEDS IMPROVEMENT"
            bar = "#" * int(val * 20) if val == val else "timeout"
            print(f"  {col:<25} {val:.3f}  {bar:<20} [{status}]")

    os.makedirs("data", exist_ok=True)
    with open("data/ragas_results.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    print("\nSaved -> data/ragas_results.json")

    _print_improvement_tips(result_dict)
    return result_dict


def _print_improvement_tips(results: dict):
    """Print targeted improvement tips based on which scores are low."""
    print("\n=== HOW TO IMPROVE SCORES ===")

    faith = results.get("faithfulness",     0)
    rel   = results.get("answer_relevancy", 0)
    prec  = results.get("context_precision",0)

    if faith < 0.7:
        print(
            "  Faithfulness low: LLM is adding info not in retrieved chunks.\n"
            "  Fix 1: Add this line to PROMPT_TEMPLATE in rag.py:\n"
            "         'Only use the context. Never add from your training data.'\n"
            "  Fix 2: Set temperature=0.0 in rag.py\n"
            "  Fix 3: Switch to llama3.1:8b (1b hallucinates more)"
        )

    if rel < 0.7:
        print(
            "  Answer Relevancy low: answers are vague or off-topic.\n"
            "  Fix 1: Increase k from 5 to 8 in rag.py (search_kwargs={'k': 8})\n"
            "  Fix 2: Add more TSB records via create_sample_data.py\n"
            "  Fix 3: Switch to llama3.1:8b"
        )

    if prec < 0.7:
        print(
            "  Context Precision low: retrieved chunks are not relevant enough.\n"
            "  Fix 1: Reduce chunk_size from 800 to 500 in build_vectorstore.py\n"
            "  Fix 2: Rebuild vector store: python -m src.build_vectorstore\n"
            "  Fix 3: Ensure document_text starts with Make/Model/Year/DTC fields"
        )

    if all(v >= 0.7 for v in [faith, rel, prec]):
        print("  All metrics >= 0.7. Strong baseline for llama3.2:1b.")

    print(
        "\n  Universal improvements:\n"
        "  - Switch LLM: ollama pull llama3.1:8b  (biggest single gain)\n"
        "  - Use real NHTSA data: python -m src.data_prep (10,000+ records)\n"
        "  - Rebuild vector store after any data change"
    )


if __name__ == "__main__":
    run_evaluation()