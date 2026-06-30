"""Retrieval side of the RAG pipeline.

Given a natural-language question, embed it with the *same* embedding function used at
ingestion time and run a nearest-neighbour query against the ChromaDB collection. Using
an identical embedding model is critical: the stored vectors were produced by Chroma's
DefaultEmbeddingFunction (all-MiniLM-L6-v2) in main.py, so the query vector must come from
that same model or cosine distances would be meaningless. We pass `query_embeddings`
explicitly rather than `query_texts` so retrieval never silently falls back to a different
default embedder.
"""

from chromadb.utils import embedding_functions
from langchain_core.documents import Document

from .get_rag_collection import get_rag_collection


# Reuse one embedding function instance: constructing it loads the all-MiniLM-L6-v2 model,
# so we build it once at import time rather than per query.
_embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def retrieve_chunks(question: str, k: int) -> list[Document]:
    """Return the `k` chunks most semantically similar to `question`.

    Each returned Document carries the chunk's stored metadata (source path, heading path)
    plus a `distance` field so callers can optionally surface retrieval scores.
    """
    collection = get_rag_collection()

    query_embedding = _embedding_fn([question])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns a list-of-lists keyed by query; we issued a single query so index [0].
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: list[Document] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        merged_metadata = dict(metadata or {})
        merged_metadata["distance"] = distance
        chunks.append(Document(page_content=content, metadata=merged_metadata))

    return chunks


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
