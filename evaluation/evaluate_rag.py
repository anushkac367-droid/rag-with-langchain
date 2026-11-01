import json
from pathlib import Path

from openai import AsyncOpenAI

from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from rag import ask_question


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "test_dataset.json"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3"


# ---------------------------------------------------------
# Load test dataset
# ---------------------------------------------------------

def load_test_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Create RAGAS evaluator LLM
# ---------------------------------------------------------

def create_evaluator_llm():

    client = AsyncOpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
    )

    evaluator_llm = llm_factory(
        OLLAMA_MODEL,
        provider="openai",
        client=client,
    )

    return evaluator_llm


# ---------------------------------------------------------
# Run RAG on test questions
# ---------------------------------------------------------

def build_evaluation_dataset(test_cases):

    samples = []

    for index, test_case in enumerate(test_cases, start=1):

        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"\n[{index}/{len(test_cases)}] {question}")

        result = ask_question(
            question=question,
            top_k=5,
        )

        answer = result["answer"]
        contexts = result["contexts"]

        print(f"Answer: {answer}")
        print(f"Retrieved contexts: {len(contexts)}")

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=ground_truth,
            retrieved_contexts=contexts,
        )

        samples.append(sample)

    return EvaluationDataset(
        samples=samples
    )


# ---------------------------------------------------------
# Evaluate RAG
# ---------------------------------------------------------

def evaluate_rag():

    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    # Load questions
    test_cases = load_test_dataset()

    print(f"\nLoaded {len(test_cases)} test questions.")

    # Create evaluator LLM
    evaluator_llm = create_evaluator_llm()

    # Build RAGAS dataset
    evaluation_dataset = build_evaluation_dataset(
        test_cases
    )

    # -----------------------------------------------------
    # Define metrics
    # -----------------------------------------------------

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm
    )

    context_precision = ContextPrecision(
        llm=evaluator_llm
    )

    context_recall = ContextRecall(
        llm=evaluator_llm
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # -----------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------

    print("\nRunning RAGAS evaluation...\n")

    results = []

    for sample in evaluation_dataset.samples:

        print("-" * 60)
        print(f"Question: {sample.user_input}")

        row = {
            "question": sample.user_input,
        }

        # Faithfulness
        faithfulness_result = faithfulness.score(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )

        row["faithfulness"] = faithfulness_result.value

        # Answer Relevancy
        answer_relevancy_result = answer_relevancy.score(
            user_input=sample.user_input,
            response=sample.response,
        )

        row["answer_relevancy"] = (
            answer_relevancy_result.value
        )

        # Context Precision
        context_precision_result = context_precision.score(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )

        row["context_precision"] = (
            context_precision_result.value
        )

        # Context Recall
        context_recall_result = context_recall.score(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )

        row["context_recall"] = (
            context_recall_result.value
        )

        results.append(row)

        print(
            f"Faithfulness:      {row['faithfulness']:.4f}"
        )

        print(
            f"Answer Relevancy:  "
            f"{row['answer_relevancy']:.4f}"
        )

        print(
            f"Context Precision: "
            f"{row['context_precision']:.4f}"
        )

        print(
            f"Context Recall:    "
            f"{row['context_recall']:.4f}"
        )

    # -----------------------------------------------------
    # Print final averages
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if results:

        average_faithfulness = sum(
            r["faithfulness"]
            for r in results
        ) / len(results)

        average_answer_relevancy = sum(
            r["answer_relevancy"]
            for r in results
        ) / len(results)

        average_context_precision = sum(
            r["context_precision"]
            for r in results
        ) / len(results)

        average_context_recall = sum(
            r["context_recall"]
            for r in results
        ) / len(results)

        print(
            f"\nFaithfulness:      "
            f"{average_faithfulness:.4f}"
        )

        print(
            f"Answer Relevancy:  "
            f"{average_answer_relevancy:.4f}"
        )

        print(
            f"Context Precision: "
            f"{average_context_precision:.4f}"
        )

        print(
            f"Context Recall:    "
            f"{average_context_recall:.4f}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate_rag()