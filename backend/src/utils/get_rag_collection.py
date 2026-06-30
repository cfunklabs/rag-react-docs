from pathlib import Path

import chromadb

from .config import load_pyproject


DATASTORE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_datastore"
RAG_COLLECTION_NAME = load_pyproject()["tool"]["rag_db"]["rag_doc_collection_name"]


def get_rag_collection():
    """Return the persistent ChromaDB collection that stores document embeddings.

    Both the ingestion pipeline and the retrieval pipeline import this helper so they
    point at the same datastore path and collection name without duplicating the wiring.
    """
    client = chromadb.PersistentClient(path=str(DATASTORE_DIR))
    return client.get_collection(name=RAG_COLLECTION_NAME)


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
