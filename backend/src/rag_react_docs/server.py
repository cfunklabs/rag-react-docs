"""MCP server exposing the RAG retrieval pipeline over stdio.

Published as the `cfunklabs-rag-react-docs` console script (`uvx cfunklabs-rag-react-docs`). It
exposes a single `search_react_docs` tool that performs *retrieval only* against the downloaded
ChromaDB collection and returns the top-k chunks with their source/heading labels. The
consuming LLM (Cursor, Claude Desktop, etc.) ingests those chunks and generates its own
grounded answer, so no Anthropic API key or generation stack is needed server-side.
"""

import sys
from typing import TypedDict

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .config import DEFAULT_TOP_K, INDEX_URL, INDEX_VERSION, REACT_VERSION_LABEL
from .datastore import get_rag_collection
from .retrieval import retrieve_chunks


# Surfaced to MCP clients as the server's usage guidance. Kept prescriptive so a consuming LLM
# reaches for this tool instead of relying on its own (possibly stale) React knowledge.
SERVER_INSTRUCTIONS = f"""\
Semantic search over the official React documentation (React {REACT_VERSION_LABEL}, index \
{INDEX_VERSION}).

Use the `search_react_docs` tool whenever a task touches React itself -- hooks, built-in \
components, APIs, rendering/effects behavior, or idiomatic patterns -- instead of answering \
from the model's own training data, which may be stale or version-mismatched. It is the \
authoritative source for React {REACT_VERSION_LABEL} in this session.

The server is retrieval-only: `search_react_docs` returns ranked documentation chunks and the \
client composes the grounded answer. Always cite the `source` label of each chunk you rely on \
so the user can trace claims back to the docs."""


mcp = FastMCP("rag-react-docs", instructions=SERVER_INSTRUCTIONS)


class SearchResult(TypedDict):
    """One retrieved documentation chunk.

    - source:   human-readable provenance label (file path > heading path) for citation
    - content:  the raw chunk text to ground an answer on
    - distance: retrieval distance (squared L2 over normalized embeddings; lower is more similar)
    """

    source: str
    content: str
    distance: float | None


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


# Passed as the tool `description` (an f-string, so version numbers interpolate -- a plain
# docstring can't). FastMCP uses this over the function docstring when both are present.
_SEARCH_DESCRIPTION = f"""\
Semantically search the official React documentation and return the most relevant chunks.

When to use: reach for this on ANY question about React itself -- hooks (`useState`, \
`useEffect`, ...), built-in components, APIs, rendering/effects/StrictMode behavior, migration, \
or idiomatic patterns. Prefer it over answering from memory: it indexes React \
{REACT_VERSION_LABEL} (index `{INDEX_VERSION}`), so it is more current and authoritative than \
the model's own training data. It does NOT cover unrelated topics or third-party libraries.

Args:
    question: A full natural-language question, not bare keywords -- richer phrasing retrieves
        better.
        Good: "How do I run cleanup logic when a component unmounts with useEffect?"
        Good: "What's the difference between useMemo and useCallback?"
        Weak: "useEffect" (too terse; ambiguous intent)
    k: How many chunks to return (default {DEFAULT_TOP_K}). Suggested by intent: ~3 for a
        specific API/signature lookup, {DEFAULT_TOP_K} for a general question, ~8-10 for broad
        or exploratory topics that likely span multiple doc pages.

Returns a list of results ordered by relevance (closest first). Each item has:
    - source:   human-readable provenance label (file path > heading path); cite this.
    - content:  the raw chunk text to ground an answer on.
    - distance: retrieval distance -- squared L2 over normalized embeddings, so LOWER is more
        similar. As a rough guide for this corpus: < ~1.0 is relevant, and > ~1.5 usually means
        the question is off-topic or not covered (no strong match). If every result is above
        ~1.5, prefer saying the docs don't cover it over guessing."""


@mcp.tool(
    name="search_react_docs",
    title="Search React Documentation",
    description=_SEARCH_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
def search_react_docs(question: str, k: int = DEFAULT_TOP_K) -> list[SearchResult]:
    """Retrieve the top-k React-docs chunks for `question` (see tool description for guidance)."""
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
