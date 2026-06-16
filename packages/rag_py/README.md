# rag_py

Python tooling for the RAG demo, built with LangChain, LangGraph, and Anthropic Claude.

## Prerequisites

- [Python 3.14+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — used for dependency management and running the project
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

All commands should be run from the `packages/rag_py` directory.

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

### 3. Fetch the React docs dataset

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
