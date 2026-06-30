# rag-backend

Backend for the RAG demo, built with LangChain, LangGraph, Anthropic Claude, and ChromaDB.

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
