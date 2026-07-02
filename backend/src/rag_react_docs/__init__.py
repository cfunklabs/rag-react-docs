"""rag-react-docs: a retrieval-only MCP server over the indexed React documentation.

Distributed on PyPI as `cfunklabs-rag-react-docs`; imported as `rag_react_docs`. The prebuilt
ChromaDB index is downloaded from a GitHub Release on first run (see `datastore.py`), so end
users never run the ingestion pipeline themselves.
"""

__version__ = "0.1.3"
