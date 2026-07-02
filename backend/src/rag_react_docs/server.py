"""MCP server exposing the RAG retrieval pipeline over stdio.

Published as the `cfunklabs-rag-react-docs` console script (`uvx cfunklabs-rag-react-docs`). It
exposes a single `search_docs` tool that performs *retrieval only* against the downloaded
ChromaDB collection and returns the top-k chunks with their source/heading labels. The
consuming LLM (Cursor, Claude Desktop, etc.) ingests those chunks and generates its own
grounded answer, so no Anthropic API key or generation stack is needed server-side.
"""

import sys

from mcp.server.fastmcp import FastMCP

from .config import DEFAULT_TOP_K
from .datastore import get_rag_collection
from .retrieval import retrieve_chunks


mcp = FastMCP("rag-react-docs")


def _collection_is_empty() -> bool:
    try:
        return get_rag_collection().count() == 0
    except Exception:
        # Treat a missing/uninitialized/failed-download collection the same as an empty one.
        return True


@mcp.tool()
def search_docs(question: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search the indexed React documentation and return the most relevant chunks.

    Args:
        question: A natural-language question to search the React docs for.
        k: How many chunks to return (defaults to `DEFAULT_TOP_K`).

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
                    "The documentation index is empty or could not be loaded. "
                    "Check network access on first run so the index can be downloaded."
                ),
                "distance": None,
            }
        ]

    return retrieve_chunks(question, k)


def main() -> None:
    """Console-script entry point: start the MCP server on stdio."""
    # Human-facing messages must go to stderr: the stdio transport reserves stdout for the
    # JSON-RPC protocol, so anything printed there would corrupt the stream.
    print(f"[rag-react-docs] MCP server starting on stdio (top_k={DEFAULT_TOP_K}).", file=sys.stderr)
    print("[rag-react-docs] Ensuring documentation index is available...", file=sys.stderr)
    try:
        if _collection_is_empty():
            print(
                "[rag-react-docs] Warning: index empty or unavailable -- check network access.",
                file=sys.stderr,
            )
        else:
            print("[rag-react-docs] Index ready.", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive; _collection_is_empty swallows most
        print(f"[rag-react-docs] Warning: could not verify index: {exc}", file=sys.stderr)

    print("[rag-react-docs] Ready. Press Ctrl+C to stop.", file=sys.stderr)
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n[rag-react-docs] Shutting down.", file=sys.stderr)


if __name__ == "__main__":
    main()
