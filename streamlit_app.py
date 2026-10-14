import json
import os
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


@st.cache_resource
def load_vector_store():
    """Load the saved FAISS index and its matching chunk metadata once."""
    loaded_index = faiss.read_index(
        str(storage_directory / "novatech.index")
    )

    with open(
        storage_directory / "metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        loaded_metadata = json.load(file)

    if loaded_index.ntotal != len(loaded_metadata):
        raise ValueError(
            "The FAISS index and metadata contain different numbers of records."
        )

    return loaded_index, loaded_metadata


try:
    index, chunk_metadata = load_vector_store()
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(f"Could not load the vector store: {error}")
    st.stop()


def retrieve_chunks(question, top_k):
    """Embed a question and return its most similar knowledge-base chunks."""
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
    return response.output_text


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

if st.session_state.get("active_question") != question:
    st.session_state["active_question"] = question
    st.session_state.pop("without_rag_answer", None)
    st.session_state.pop("with_rag_answer", None)
    st.session_state.pop("retrieved_chunks", None)

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
            with st.spinner("Retrieving evidence and generating an answer..."):
                retrieved_chunks = retrieve_chunks(question, top_k)
                st.session_state["retrieved_chunks"] = retrieved_chunks

                if retrieved_chunks:
                    st.session_state["with_rag_answer"] = generate_with_rag(
                        question,
                        retrieved_chunks,
                    )
                else:
                    st.session_state["with_rag_answer"] = (
                        "I don't have enough information in the knowledge base."
                    )
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

st.divider()
st.subheader("Retrieved Context")

retrieved_chunks = st.session_state.get("retrieved_chunks", [])

if not retrieved_chunks:
    st.caption("Retrieved filenames, similarity scores, and text appear here.")
else:
    for rank, result in enumerate(retrieved_chunks, start=1):
        label = (
            f"#{rank}  {result['source']}  ·  "
            f"score {result['score']:.4f}"
        )
        with st.expander(label, expanded=(rank == 1)):
            st.caption(f"Chunk {result['chunk_id']}")
            st.write(result["text"])
