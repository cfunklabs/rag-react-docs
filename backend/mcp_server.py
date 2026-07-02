"""MCP server exposing the RAG retrieval pipeline over stdio.

This is the second entry point into the system (alongside the `query.py` CLI). It exposes a
single `search_docs` tool that performs *retrieval only* against the ChromaDB collection and
returns the top-k chunks with their source/heading labels. The consuming LLM (Cursor, Claude
Desktop, etc.) ingests those chunks and generates its own grounded answer, so no Anthropic API
key or generation stack is needed server-side -- retrieval only touches the local embedding
model and the vector store.

Run it from the `backend/` directory so `load_rag_query_config()` resolves `pyproject.toml`:

    uv run mcp_server.py
"""

import sys

from mcp.server.fastmcp import FastMCP

from src.utils.config import load_rag_query_config
from src.utils.format_source_label import format_source_label
from src.utils.get_rag_collection import get_rag_collection
from src.utils.retrieve_chunks import retrieve_chunks


mcp = FastMCP("rag-react-docs")

DEFAULT_TOP_K = load_rag_query_config()["top_k"]


def _collection_is_empty() -> bool:
    try:
        return get_rag_collection().count() == 0
    except Exception:
        # Treat a missing/uninitialized collection the same as an empty one: the user needs
        # to run the ingestion pipeline before retrieval can return anything useful.
        return True


@mcp.tool()
def search_docs(question: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search the indexed React documentation and return the most relevant chunks.

    Args:
        question: A natural-language question to search the React docs for.
        k: How many chunks to return (defaults to `top_k` from pyproject.toml).

    Returns a list of results ordered by relevance. Each item has:
        - source:   a human-readable provenance label (file path > heading path)
        - content:  the raw chunk text to ground an answer on
        - distance: the retrieval distance (lower is more similar)
    """
    if _collection_is_empty():
        return [
            {
                "source": "rag-react-docs",
                "content": (
                    "The documentation index is empty or uninitialized. Run "
                    "'uv run main.py' from the backend directory to ingest the docs "
                    "before searching."
                ),
                "distance": None,
            }
        ]

    chunks = retrieve_chunks(question, k)
    return [
        {
            "source": format_source_label(chunk.metadata),
            "content": chunk.page_content,
            "distance": chunk.metadata.get("distance"),
        }
        for chunk in chunks
    ]


if __name__ == "__main__":
    # Human-facing messages must go to stderr: the stdio transport reserves stdout for the
    # JSON-RPC protocol, so anything printed there would corrupt the stream.
    print(f"[rag-react-docs] MCP server starting on stdio (top_k={DEFAULT_TOP_K}).", file=sys.stderr)
    if _collection_is_empty():
        print("[rag-react-docs] Warning: index empty -- run 'uv run main.py' first.", file=sys.stderr)
    print("[rag-react-docs] Ready. Press Ctrl+C to stop.", file=sys.stderr)
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n[rag-react-docs] Shutting down.", file=sys.stderr)
