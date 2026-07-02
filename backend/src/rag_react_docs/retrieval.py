"""Retrieval side of the RAG pipeline for the published package.

Embeds the question with the *same* embedding function used at ingestion (Chroma's
DefaultEmbeddingFunction, all-MiniLM-L6-v2) and runs a nearest-neighbour query against the
downloaded ChromaDB collection. Using an identical embedding model is critical: the stored
vectors were produced by that same model, so a different embedder would make cosine distances
meaningless. Results are returned as plain dicts so the package needs no langchain-core.
"""

from chromadb.utils import embedding_functions

from .datastore import get_rag_collection
from .source_label import format_source_label


# Constructing the embedding function loads the all-MiniLM-L6-v2 model, so build it once at
# import time rather than per query.
_embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def retrieve_chunks(question: str, k: int) -> list[dict]:
    """Return the `k` chunks most semantically similar to `question`.

    Each result is a dict with `source` (provenance label), `content` (raw chunk text), and
    `distance` (retrieval distance; lower is more similar).
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

    return [
        {
            "source": format_source_label(dict(metadata or {})),
            "content": content,
            "distance": distance,
        }
        for content, metadata, distance in zip(documents, metadatas, distances)
    ]
