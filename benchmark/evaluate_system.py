import argparse
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-5.4-mini"
NOT_FOUND_MESSAGE = "I don't have enough information in the knowledge base."

project_directory = Path(__file__).resolve().parents[1]
benchmark_directory = project_directory / "benchmark"
vector_store_directory = project_directory / "vector_store"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalized_terms(text):
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "can",
        "in",
        "is",
        "no",
        "of",
        "or",
        "the",
        "to",
        "unless",
        "wherever",
    }
    normalized_text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())

    return {
        term
        for term in re.findall(r"[a-z0-9]+", normalized_text)
        if term not in stop_words
    }


def lexical_answer_match(expected_answer, generated_answer):
    """Return a transparent, approximate correctness signal."""
    expected_terms = normalized_terms(expected_answer)
    answer_terms = normalized_terms(generated_answer)

    if not expected_terms:
        return False

    coverage = len(expected_terms & answer_terms) / len(expected_terms)
    return coverage >= 0.6


def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        (
            f"[Source: {chunk['source']}, Chunk: {chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )
        for chunk in retrieved_chunks
    )

    return f"""
You are a NovaTech company knowledge assistant.

Answer the question using only the provided context.
If the answer cannot be found in the context, respond exactly:
"{NOT_FOUND_MESSAGE}"

Include the source filename used to answer the question.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main(top_k):
    client = OpenAI()
    questions = load_json(benchmark_directory / "questions.json")
    metadata = load_json(vector_store_directory / "metadata.json")
    index = faiss.read_index(str(vector_store_directory / "novatech.index"))

    if index.ntotal != len(metadata):
        raise ValueError(
            "The FAISS index and metadata contain different record counts."
        )

    result_count = min(max(1, top_k), index.ntotal)

    rows = []
    evidence_successes = 0
    answer_successes = 0
    grounded_successes = 0
    abstention_successes = 0
    answerable_count = 0
    unanswerable_count = 0
    total_latency = 0.0

    for item in questions:
        started = time.perf_counter()
        retrieved_chunks = []

        embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=item["question"],
        )
        query_vector = np.array(
            [embedding_response.data[0].embedding],
            dtype="float32",
        )
        faiss.normalize_L2(query_vector)
        scores, positions = index.search(query_vector, result_count)

        for score, position in zip(
            scores[0],
            positions[0],
        ):
            if position == -1:
                continue

            chunk = metadata[position]
            retrieved_chunks.append(
                {
                    "score": float(score),
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
            )

        response = client.responses.create(
            model=GENERATION_MODEL,
            input=build_prompt(item["question"], retrieved_chunks),
        )
        answer = response.output_text
        latency = time.perf_counter() - started
        total_latency += latency

        retrieved_sources = {
            chunk["source"]
            for chunk in retrieved_chunks
        }
        expected_evidence = item["evidence"]
        is_unanswerable = expected_evidence is None

        if is_unanswerable:
            unanswerable_count += 1
            evidence_found = None
            abstained = NOT_FOUND_MESSAGE.lower() in answer.lower()
            answer_correct = abstained
            grounded = abstained
            abstention_successes += int(abstained)
        else:
            answerable_count += 1
            evidence_found = expected_evidence in retrieved_sources
            answer_correct = lexical_answer_match(
                item["expected_answer"],
                answer,
            )
            grounded = evidence_found and answer_correct
            evidence_successes += int(evidence_found)

        answer_successes += int(answer_correct)
        grounded_successes += int(grounded)

        rows.append(
            {
                "Question": item["question"],
                "Evidence Found": (
                    "N/A" if evidence_found is None else "✓" if evidence_found else "✗"
                ),
                "Correct Answer": "✓" if answer_correct else "✗",
                "Grounded": "✓" if grounded else "✗",
                "Latency": f"{latency:.2f}s",
                "Expected Source": expected_evidence or "N/A",
                "Answer": answer,
            }
        )

    summary = {
        "top_k": result_count,
        "evidence_recall": safe_rate(evidence_successes, answerable_count),
        "answer_accuracy": safe_rate(answer_successes, len(questions)),
        "groundedness": safe_rate(grounded_successes, len(questions)),
        "abstention_accuracy": safe_rate(
            abstention_successes,
            unanswerable_count,
        ),
        "average_latency_seconds": safe_rate(total_latency, len(questions)),
    }

    output = {
        "evaluation_note": (
            "Correctness uses a 60% lexical coverage heuristic. "
            "Groundedness requires both expected evidence and that heuristic."
        ),
        "rows": rows,
        "summary": summary,
    }

    output_path = benchmark_directory / "results.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    arguments = parser.parse_args()
    main(arguments.top_k)
