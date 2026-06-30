"""Build human-readable provenance labels from a chunk's stored metadata.

Kept in its own module (rather than inside rag_graph.py) so retrieval-only callers such as
the MCP server can cite sources without importing the generation stack (ChatAnthropic et al.).
"""

from pathlib import Path


_DOCS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "docs"


def format_source_label(metadata: dict) -> str:
    """Build a human-readable provenance label from a chunk's metadata.

    Combines the source file (shortened to a path relative to the docs directory when
    possible) with the stored heading hierarchy, e.g.
    "API Reference/React/APIs/memo.md > Reference > memo(Component, arePropsEqual?)".
    """
    source = metadata.get("source", "unknown source")
    try:
        source = str(Path(source).resolve().relative_to(_DOCS_DIR.resolve()))
    except (ValueError, OSError):
        source = Path(source).name

    headings = [
        metadata[field]
        for field in ("Header 1", "Header 2", "Header 3")
        if metadata.get(field)
    ]
    if headings:
        return f"{source} > {' > '.join(headings)}"
    return source


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
