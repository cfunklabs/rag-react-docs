# cfunklabs-rag-react-docs

Backend for the RAG demo, built with LangChain, LangGraph, Anthropic Claude, and ChromaDB.
It also ships a retrieval-only MCP server, published to PyPI as `cfunklabs-rag-react-docs`,
that serves grounding context from the indexed React documentation (see [MCP server](#mcp-server)).

## Prerequisites

- [Python 3.14+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — used for dependency management and running the project
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

All commands should be run from the `backend` directory.

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Copy the sample env file and add your Anthropic API key:

```bash
cp .env.sample .env
```

Open `.env` and set your key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

### 3. Initialize the vector database

Create the ChromaDB collection used to store document embeddings:

```bash
uv run src/utils/init_db.py
```

This creates a persistent ChromaDB store under `rag_datastore/` and a collection
named after `tool.rag_db.rag_doc_collection_name` in [pyproject.toml](pyproject.toml).
The `rag_datastore/` directory is gitignored.

### 4. Fetch the React docs dataset

The corpus is a local mirror of the [React documentation](https://react.dev/) markdown files. Download it with:

```bash
uv run src/utils/fetch_react_docs.py
```

This script:

1. Fetches the index at [react.dev/llms.txt](https://react.dev/llms.txt)
2. Follows every linked `https://react.dev/*.md` URL
3. Saves each page under `docs/` at the **repository root**, using the index heading structure as directories and each file's frontmatter `title` as the filename

Example output path:

```
docs/API Reference/React/Components/Built-in React Components.md
```

The `docs/` directory is gitignored — run the script locally after cloning and re-run it anytime to refresh the dataset.

## Running

### Startup check

Run `main.py` to verify your environment is configured correctly. It prints installed package versions and performs a live health check against the Claude API:

```bash
uv run main.py
```

A passing run looks like:

```
Versions:
  - langchain_core_version: x.x.x
  - langgraph_version: x.x.x
  - langchain_anthropic_version: x.x.x

Checking LLM service health... PASSED
```

If the health check fails, confirm that `ANTHROPIC_API_KEY` is set correctly in your `.env` file.

### Ingesting documents

Chunk every Markdown file under `docs/`, embed the chunks, and upsert them into the vector store:

```bash
uv run main.py
```

To process a single file (useful while iterating on chunking), pass `--md_file_path`. Add
`--evaluate_chunking` to write LLM-as-judge quality reports to `evals/results`, or
`--print_chunks` to dump each chunk to stdout.

### Querying

Ask a question against the ingested docs. The query pipeline is a LangGraph graph
(`retrieve` -> `generate`) that embeds your question with the same model used at ingestion,
retrieves the most similar chunks from ChromaDB, and has Claude answer using only that
context. The answer streams token-by-token and is followed by the cited sources:

```bash
uv run query.py "How does memo work?"
```

Options:

- `--k N` — number of chunks to retrieve (default `top_k` in [pyproject.toml](pyproject.toml)).
- `--no-stream` — wait for the full answer instead of streaming tokens.
- `--show-scores` — show the retrieval distance for each cited source.
- `-v`, `--verbose` — show diagnostic output (the preflight datastore and LLM health checks). Hidden by default.

Retrieval and generation settings live under `[tool.rag_query]` in [pyproject.toml](pyproject.toml)
(`top_k` and `generation_model`). Run `uv run main.py` first — querying requires a populated
collection.

### MCP server

In addition to the CLI, the retrieval pipeline is exposed as an [MCP](https://modelcontextprotocol.io/)
server over stdio, so MCP clients (Cursor, Claude Desktop, etc.) can pull grounding context
directly. The server ships prescriptive metadata — a server-level instructions block plus a
richly documented tool — so a consuming LLM knows when to reach for it (any React 19.2 API,
hook, component, or pattern question) instead of relying on its own possibly-stale knowledge.
It exposes a single **retrieval-only** tool:

- `search_react_docs(question, k?)` — embeds the question with the same model used at ingestion,
  retrieves the most similar chunks from ChromaDB, and returns each chunk's `source` label,
  `content`, and retrieval `distance`. The client LLM generates the answer from those chunks,
  so no Anthropic key is needed to run the server.
  - `question` should be a full natural-language question, not bare keywords.
  - `k` defaults to `RAG_TOP_K` (5); use ~3 for a specific API lookup and ~8-10 for broad topics.
  - `distance` is squared L2 over normalized embeddings, so **lower is more similar**. For this
    corpus, `< ~1.0` is relevant and `> ~1.5` usually means off-topic / not covered.

#### For end users (published package)

The server is published to PyPI as **`cfunklabs-rag-react-docs`**. End users don't clone the
repo or run the ingestion pipeline — the prebuilt index (~34 MB) is downloaded from a GitHub
Release and cached on first run. Just register it with your MCP client:

```json
{
  "mcpServers": {
    "rag-react-docs": {
      "command": "uvx",
      "args": ["cfunklabs-rag-react-docs"]
    }
  }
}
```

The first launch needs network access to fetch the index; subsequent runs read from the local
cache (`platformdirs` cache dir) and work offline. Optional environment overrides: `RAG_TOP_K`,
`RAG_COLLECTION_NAME`, `RAG_INDEX_URL`, and `RAG_DATASTORE_DIR`.

#### Local development

Run the dev server from the `backend` directory (so `pyproject.toml` resolves) against the
locally-built `rag_datastore`:

```bash
uv run mcp_server.py
```

Run standalone this way, the server prints a short startup banner to stderr and then blocks
silently by design — the stdio transport reserves stdout for the JSON-RPC protocol, so it
waits for a client to connect rather than logging. Running it directly is mainly a smoke test;
press Ctrl+C to stop.

Run `uv run main.py` first — the dev server needs a populated collection. For interactive
testing, launch the MCP Inspector with `uv run mcp dev mcp_server.py`.

### Publishing to PyPI

The published package (`cfunklabs-rag-react-docs`) contains only the retrieval + MCP server
(the import package `rag_react_docs` under `src/`). Ingestion/query tooling and `src/utils/*`
are dev-only and excluded from the wheel.

Two artifacts get published: the Python package (to PyPI) and the prebuilt index (to a GitHub
Release). They version independently — the index version is pinned as `INDEX_VERSION` in
[src/rag_react_docs/config.py](src/rag_react_docs/config.py).

1. Build and upload the index archive (after `uv run main.py` has populated `rag_datastore`):

```bash
uv run scripts/build_index_archive.py
gh release create index-19-2-v1 dist/rag-index-19-2-v1.tar.gz dist/rag-index-19-2-v1.tar.gz.sha256
```

2. Build and publish the package (test on TestPyPI first):

```bash
uv build                                   # -> dist/ wheel + sdist (only rag_react_docs)
uv publish --publish-url https://test.pypi.org/legacy/   # TestPyPI dry run
uv publish                                 # PyPI
```

The index version follows the standard `index-<react-version>-v<incremental>` (e.g.
`index-19-2-v1`), composed in [src/rag_react_docs/config.py](src/rag_react_docs/config.py) from
`REACT_VERSION` and `INDEX_REVISION`. Bump `REACT_VERSION` when re-fetching the docs for a new
React release, and bump `INDEX_REVISION` for re-chunk or embedding-model changes within the same
React version. Either bump changes the release tag/asset name and cache path, so clients pull a
fresh, compatible index instead of reusing a stale cache — re-release the archive under the new
`index-<react-version>-v<incremental>` tag.
