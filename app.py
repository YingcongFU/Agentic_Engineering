from openai import OpenAI
from pathlib import Path

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4-mini",
    instructions="You are a helpful AI assistant.",
    input="How long do NovaTech Pro customers have to request a refund?",
)

##Loading knowledge base
knowledge_base = Path(__file__).resolve().parent/"knowledge_base"
documents = []
for file in knowledge_base.glob("*.md"):
    text = file.read_text(encoding="utf-8")

    documents.append({
        "source": file.name,
        "text": text
    })

##Chunking
def chunk_text(text, chunk_size=500, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

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


chunks = []

for document in documents:
    document_chunks = chunk_text(
        document["text"],
        chunk_size=500,
        overlap=50
    )

    for chunk_number, chunk in enumerate(document_chunks):
        chunks.append({
            "source": document["source"],
            "chunk_id": chunk_number,
            "text": chunk
        })

##Embedding
if not chunks:
    raise ValueError("No chunks were created")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[chunk["text"] for chunk in chunks]
)

for result in response.data:
    chunks[result.index]["embedding"] = result.embedding

print(f"Generated embeddings for {len(chunks)} chunks")
print(f"Embedding dimensions: {len(chunks[0]['embedding'])}")

#Store Embeddings

import faiss
import numpy as np

if not chunks:
    raise ValueError("No chunks are available")

if any("embedding" not in chunk for chunk in chunks):
    raise ValueError("Some chunks do not have embeddings")

# Convert the embeddings into a float32 matrix
vectors = np.array(
    [chunk["embedding"] for chunk in chunks],
    dtype="float32"
)

# Normalize vectors so inner product acts like cosine similarity
faiss.normalize_L2(vectors)

# Number of values in each embedding vector
dimension = vectors.shape[1]

# Create the FAISS index
index = faiss.IndexFlatIP(dimension)

# Store all vectors in the index
index.add(vectors)

# Keep searchable metadata in the same order as the vectors
chunk_metadata = [
    {
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"]
    }
    for chunk in chunks
]

print("Vector dimensions:", dimension)
print("Vectors stored:", index.ntotal)
print("Metadata records:", len(chunk_metadata))

#Save the index
import json
from pathlib import Path

storage_directory = Path(__file__).resolve().parent / "vector_store"
storage_directory.mkdir(exist_ok=True)

faiss.write_index(
    index,
    str(storage_directory / "novatech.index")
)

with open(
    storage_directory / "metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(chunk_metadata, file, indent=2)

print("Vector index saved to:", storage_directory)

#Embed User Question
question = input("Ask a question: ").strip()

if not question:
    raise ValueError("Question cannot be empty")

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

query_embedding = query_response.data[0].embedding

query_vector = np.array(
    [query_embedding],
    dtype="float32"
)

faiss.normalize_L2(query_vector)

print("Question:", question)
print("Query vector shape:", query_vector.shape)
print("First five values:", query_vector[0][:5])

#Vector search
top_k = min(3, index.ntotal)

scores, positions = index.search(
    query_vector,
    top_k
)

search_results = []

for score, position in zip(scores[0], positions[0]):
    # FAISS may return -1 when no matching position exists
    if position == -1:
        continue

    metadata = chunk_metadata[position]

    search_results.append({
        "score": float(score),
        "source": metadata["source"],
        "chunk_id": metadata["chunk_id"],
        "text": metadata["text"]
    })

print(f"Retrieved {len(search_results)} chunks")

#Print retrieved chunks
print("\n" + "=" * 60)
print("QUESTION")
print(question)

print("\nRETRIEVED CHUNKS")

if not search_results:
    print("No relevant chunks were found.")

for rank, result in enumerate(search_results, start=1):
    print("\n" + "-" * 60)
    print(f"Result #{rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print("\nText:")
    print(result["text"])

print("\n" + "=" * 60)

#Augmented prompt
if not search_results:
    print("I don't have enough information in the knowledge base.")

else:
    context_parts = []

    for result in search_results:
        context_parts.append(
            f"[Source: {result['source']}, "
            f"Chunk: {result['chunk_id']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    augmented_prompt = f"""
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

    print("\n" + "=" * 60)
    print("AUGMENTED PROMPT")
    print(augmented_prompt)
    print("=" * 60)

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=augmented_prompt
    )

    answer = response.output_text

    print("\n" + "=" * 60)
    print("ANSWER")
    print(answer)
    print("=" * 60)

