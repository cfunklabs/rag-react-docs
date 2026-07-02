"""MCP server exposing the RAG retrieval pipeline over stdio.

Published as the `cfunklabs-rag-react-docs` console script (`uvx cfunklabs-rag-react-docs`). It
exposes a single `search_docs` tool that performs *retrieval only* against the downloaded
ChromaDB collection and returns the top-k chunks with their source/heading labels. The
consuming LLM (Cursor, Claude Desktop, etc.) ingests those chunks and generates its own
grounded answer, so no Anthropic API key or generation stack is needed server-side.
"""

import sys

from mcp.server.fastmcp import FastMCP

from .config import DEFAULT_TOP_K, INDEX_URL
from .datastore import get_rag_collection
from .retrieval import retrieve_chunks


mcp = FastMCP("rag-react-docs")


def _index_error() -> str | None:
    """Return a human-readable reason the index is unavailable, or None if it's ready.

    Distinguishes a real load/download failure (surfacing the underlying exception and the
    URL it tried) from a genuinely empty collection, so callers report the actual cause rather
    than a catch-all "index is empty" message.
    """
    try:
        count = get_rag_collection().count()
    except Exception as exc:
        return (
            f"Could not load the documentation index (downloaded from {INDEX_URL}): "
            f"{type(exc).__name__}: {exc}"
        )
    if count == 0:
        return "The documentation index loaded but contains no documents."
    return None


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
    error = _index_error()
    if error:
        print(f"[rag-react-docs] {error}", file=sys.stderr)
        return [{"source": "rag-react-docs", "content": error, "distance": None}]

    return retrieve_chunks(question, k)


def main() -> None:
    """Console-script entry point: start the MCP server on stdio."""
    # Human-facing messages must go to stderr: the stdio transport reserves stdout for the
    # JSON-RPC protocol, so anything printed there would corrupt the stream.
    print(f"[rag-react-docs] MCP server starting on stdio (top_k={DEFAULT_TOP_K}).", file=sys.stderr)
    print("[rag-react-docs] Ensuring documentation index is available...", file=sys.stderr)
    error = _index_error()
    if error:
        print(f"[rag-react-docs] Warning: {error}", file=sys.stderr)
    else:
        print("[rag-react-docs] Index ready.", file=sys.stderr)

    print("[rag-react-docs] Ready. Press Ctrl+C to stop.", file=sys.stderr)
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n[rag-react-docs] Shutting down.", file=sys.stderr)


if __name__ == "__main__":
    main()
