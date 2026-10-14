import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="NovaTech RAG Playground",
    page_icon="🔎",
    layout="wide",
)

st.title("NovaTech RAG Playground")
st.write(
    "Ask the same question with and without retrieval, then inspect the "
    "evidence used by the RAG answer."
)


def get_api_key():
    """Read the API key from Streamlit secrets or the local environment."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("OPENAI_API_KEY")


api_key = get_api_key()

if not api_key:
    st.error(
        "OPENAI_API_KEY is missing. Add it to Streamlit Cloud Secrets or "
        "set it as a local environment variable."
    )
    st.stop()

client = OpenAI(api_key=api_key)
GENERATION_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

project_directory = Path(__file__).resolve().parent
storage_directory = project_directory / "vector_store"
knowledge_base_directory = project_directory / "knowledge_base"


@st.cache_resource
def load_vector_store(chunk_size):
    """Load or build a FAISS index for the selected chunk size."""
    if chunk_size == 500:
        loaded_index = faiss.read_index(
            str(storage_directory / "novatech.index")
        )

        with open(
            storage_directory / "metadata.json",
            "r",
            encoding="utf-8",
        ) as file:
            loaded_metadata = json.load(file)
    else:
        loaded_metadata = build_chunk_metadata(chunk_size)

        if not loaded_metadata:
            raise ValueError("No Markdown content was found in knowledge_base.")

        embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[item["text"] for item in loaded_metadata],
        )

        vectors = np.array(
            [item.embedding for item in embedding_response.data],
            dtype="float32",
        )
        faiss.normalize_L2(vectors)

        loaded_index = faiss.IndexFlatIP(vectors.shape[1])
        loaded_index.add(vectors)

    if loaded_index.ntotal != len(loaded_metadata):
        raise ValueError(
            "The FAISS index and metadata contain different numbers of records."
        )

    return loaded_index, loaded_metadata


def chunk_text(text, chunk_size, overlap=50):
    """Split text into overlapping character-based chunks."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_chunk_metadata(chunk_size):
    """Load Markdown documents and split them using the selected size."""
    metadata = []

    for file_path in sorted(knowledge_base_directory.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")

        for chunk_id, text_chunk in enumerate(
            chunk_text(text, chunk_size=chunk_size)
        ):
            metadata.append(
                {
                    "source": file_path.name,
                    "chunk_id": chunk_id,
                    "text": text_chunk,
                }
            )

    return metadata


def retrieve_chunks(question, top_k, chunk_size):
    """Embed a question and return its most similar knowledge-base chunks."""
    index, chunk_metadata = load_vector_store(chunk_size)

    query_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    query_vector = np.array(
        [query_response.data[0].embedding],
        dtype="float32",
    )
    faiss.normalize_L2(query_vector)

    result_count = min(top_k, index.ntotal)
    scores, positions = index.search(query_vector, result_count)

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position == -1:
            continue

        metadata = chunk_metadata[position]
        results.append(
            {
                "score": float(score),
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "text": metadata["text"],
            }
        )

    return results


def generate_without_rag(question):
    """Send only the question to the generation model."""
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=(
            "You are a helpful assistant. Answer the question honestly. "
            "Do not claim access to NovaTech's private company policies."
        ),
        input=question,
    )
    return response.output_text


def generate_with_rag(question, search_results):
    """Generate an answer grounded only in the retrieved chunks."""
    context = "\n\n".join(
        (
            f"[Source: {result['source']}, Chunk: {result['chunk_id']}]\n"
            f"{result['text']}"
        )
        for result in search_results
    )

    prompt = f"""
You are a NovaTech company knowledge assistant.

Answer the question using only the provided context.
If the answer cannot be found in the context, respond exactly:
"I don't have enough information in the knowledge base."

Include the source filename used to answer the question.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()

    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt,
    )
    return {
        "answer": response.output_text,
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
    }


question = st.text_input(
    "Ask NovaTech",
    placeholder="How long is the Pro refund period?",
)

top_k = st.slider(
    "Top-K chunks",
    min_value=1,
    max_value=5,
    value=3,
    help="The number of knowledge-base chunks retrieved for the RAG answer.",
)

chunk_size = st.select_slider(
    "Chunk size (characters)",
    options=[200, 500, 1000],
    value=500,
    help=(
        "Changing chunk size rebuilds the document index. The 500-character "
        "index is prebuilt; other sizes are generated and cached on first use."
    ),
)

if chunk_size != 500:
    st.caption(
        "The first RAG request for this chunk size will take longer because "
        "the document embeddings must be generated once."
    )

if st.session_state.get("active_question") != question:
    st.session_state["active_question"] = question
    st.session_state.pop("without_rag_answer", None)
    st.session_state.pop("with_rag_answer", None)
    st.session_state.pop("retrieved_chunks", None)
    st.session_state.pop("rag_metrics", None)

retrieval_config = (question, top_k, chunk_size)

if st.session_state.get("retrieval_config") != retrieval_config:
    st.session_state["retrieval_config"] = retrieval_config
    st.session_state.pop("with_rag_answer", None)
    st.session_state.pop("retrieved_chunks", None)
    st.session_state.pop("retrieved_chunk_size", None)
    st.session_state.pop("rag_metrics", None)

without_button_column, with_button_column = st.columns(2)

with without_button_column:
    ask_without_rag = st.button(
        "Ask Without RAG",
        use_container_width=True,
    )

with with_button_column:
    ask_with_rag = st.button(
        "Ask With RAG",
        type="primary",
        use_container_width=True,
    )

if ask_without_rag:
    if not question.strip():
        st.warning("Enter a question before asking the model.")
    else:
        try:
            with st.spinner("Generating an answer without company context..."):
                st.session_state["without_rag_answer"] = generate_without_rag(
                    question
                )
        except Exception as error:
            st.error(f"The non-RAG request failed: {error}")

if ask_with_rag:
    if not question.strip():
        st.warning("Enter a question before searching the knowledge base.")
    else:
        try:
            total_started = time.perf_counter()

            with st.spinner(
                f"Retrieving {top_k} chunks at size {chunk_size} and "
                "generating an answer..."
            ):
                retrieval_started = time.perf_counter()
                retrieved_chunks = retrieve_chunks(
                    question,
                    top_k,
                    chunk_size,
                )
                retrieval_seconds = time.perf_counter() - retrieval_started
                st.session_state["retrieved_chunks"] = retrieved_chunks
                st.session_state["retrieved_chunk_size"] = chunk_size

                if retrieved_chunks:
                    generation_started = time.perf_counter()
                    generation_result = generate_with_rag(
                        question,
                        retrieved_chunks,
                    )
                    generation_seconds = time.perf_counter() - generation_started
                    st.session_state["with_rag_answer"] = generation_result[
                        "answer"
                    ]
                else:
                    generation_seconds = 0.0
                    generation_result = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                    st.session_state["with_rag_answer"] = (
                        "I don't have enough information in the knowledge base."
                    )

                st.session_state["rag_metrics"] = {
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": time.perf_counter() - total_started,
                    "retrieved_chunks": len(retrieved_chunks),
                    "input_tokens": generation_result["input_tokens"],
                    "output_tokens": generation_result["output_tokens"],
                    "top_score": (
                        retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
                    ),
                }
        except Exception as error:
            st.error(f"The RAG request failed: {error}")

st.divider()
without_answer_column, with_answer_column = st.columns(2)

with without_answer_column:
    st.subheader("Without RAG")
    if "without_rag_answer" in st.session_state:
        st.write(st.session_state["without_rag_answer"])
    else:
        st.caption("Select “Ask Without RAG” to establish a baseline.")

with with_answer_column:
    st.subheader("With RAG")
    if "with_rag_answer" in st.session_state:
        st.write(st.session_state["with_rag_answer"])
    else:
        st.caption("Select “Ask With RAG” to retrieve and use company context.")

rag_metrics = st.session_state.get("rag_metrics")

if rag_metrics:
    st.subheader("RAG Metrics")
    metric_columns = st.columns(7)
    metric_columns[0].metric(
        "Retrieval",
        f"{rag_metrics['retrieval_seconds'] * 1000:.0f} ms",
    )
    metric_columns[1].metric(
        "Generation",
        f"{rag_metrics['generation_seconds']:.2f} s",
    )
    metric_columns[2].metric(
        "Total",
        f"{rag_metrics['total_seconds']:.2f} s",
    )
    metric_columns[3].metric(
        "Chunks",
        rag_metrics["retrieved_chunks"],
    )
    metric_columns[4].metric(
        "Input tokens",
        rag_metrics["input_tokens"],
        help="Tokens in the full augmented prompt, including retrieved context.",
    )
    metric_columns[5].metric(
        "Output tokens",
        rag_metrics["output_tokens"],
    )
    metric_columns[6].metric(
        "Top score",
        f"{rag_metrics['top_score']:.3f}",
    )

st.divider()
st.subheader("Retrieved Context")

retrieved_chunks = st.session_state.get("retrieved_chunks", [])

if not retrieved_chunks:
    st.caption("Retrieved filenames, similarity scores, and text appear here.")
else:
    st.caption(
        "Chunk size: "
        f"{st.session_state.get('retrieved_chunk_size', chunk_size)} characters"
    )
    for rank, result in enumerate(retrieved_chunks, start=1):
        label = (
            f"#{rank}  {result['source']}  ·  "
            f"score {result['score']:.4f}"
        )
        with st.expander(label, expanded=(rank == 1)):
            st.caption(f"Chunk {result['chunk_id']}")
            st.write(result["text"])

benchmark_results_file = project_directory / "benchmark" / "results.json"

if benchmark_results_file.exists():
    with open(benchmark_results_file, "r", encoding="utf-8") as file:
        benchmark_results = json.load(file)

    st.divider()
    st.subheader("Final Benchmark")
    st.caption(
        "Correctness and groundedness are lightweight lexical heuristics; "
        "review borderline cases manually."
    )
    st.dataframe(
        benchmark_results["rows"],
        use_container_width=True,
        hide_index=True,
    )

    summary = benchmark_results["summary"]
    summary_columns = st.columns(5)
    summary_columns[0].metric(
        f"Evidence Recall@{summary['top_k']}",
        f"{summary['evidence_recall']:.1%}",
    )
    summary_columns[1].metric(
        "Answer Accuracy",
        f"{summary['answer_accuracy']:.1%}",
    )
    summary_columns[2].metric(
        "Groundedness",
        f"{summary['groundedness']:.1%}",
    )
    summary_columns[3].metric(
        "Abstention Accuracy",
        f"{summary['abstention_accuracy']:.1%}",
    )
    summary_columns[4].metric(
        "Average Latency",
        f"{summary['average_latency_seconds']:.2f} s",
    )
