"""Runtime configuration for the published package.

Unlike the dev tooling (which reads `pyproject.toml`), the installed package has no repo
checkout to read from, so defaults are baked in here and overridable via environment variables.
This keeps the wheel self-contained while still letting power users retarget the collection,
retrieval depth, index location, or download URL from their MCP client config.
"""

import os
from pathlib import Path

import platformdirs


# The all-MiniLM-L6-v2 vectors were ingested into this collection; the name must match the
# collection stored inside the downloaded index archive.
COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "rag_doc_collection")

DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# Index version standard: index-<react-version>-v<incremental> (e.g. index-19-2-v1).
# REACT_VERSION is the React docs version with dots as dashes; bump it when the corpus is
# re-fetched for a new React release. INDEX_REVISION bumps for re-chunk/embedding changes
# within the same React version. INDEX_VERSION is part of both the cache path and the release
# asset name, so any bump forces a fresh download rather than reusing a stale cached index.
REACT_VERSION = "19-2"
INDEX_REVISION = "v1"
INDEX_VERSION = f"{REACT_VERSION}-{INDEX_REVISION}"

# The prebuilt index is published as a GitHub Release asset. A sibling `<archive>.sha256` file
# is fetched alongside it to verify the download before extraction.
_DEFAULT_INDEX_URL = (
    "https://github.com/cfunklabs/rag-react-docs/releases/download/"
    f"index-{INDEX_VERSION}/rag-index-{INDEX_VERSION}.tar.gz"
)
INDEX_URL = os.environ.get("RAG_INDEX_URL", _DEFAULT_INDEX_URL)


def datastore_dir() -> Path:
    """Return the directory that holds (or will hold) the extracted ChromaDB index.

    Defaults to a per-user cache directory namespaced by index version so multiple versions
    can coexist and a version bump never collides with an older cached index. Overridable via
    `RAG_DATASTORE_DIR` (e.g. to point at a repo-local store during development).
    """
    override = os.environ.get("RAG_DATASTORE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_cache_dir("cfunklabs-rag-react-docs")) / "index" / INDEX_VERSION
