# rag_py

Python tooling for the RAG demo.

## React docs dataset

The corpus is a local mirror of the [React documentation](https://react.dev/) markdown files. Download it with `scripts/fetch_react_docs.py`, which:

1. Fetches the index at [react.dev/llms.txt](https://react.dev/llms.txt)
2. Follows every linked `https://react.dev/*.md` URL
3. Saves each page under `docs/` at the **repository root**, using the index heading structure as directories and each file's frontmatter `title` as the filename

Example output path:

```
docs/API Reference/React/Components/Built-in React Components.md
```

The `docs/` directory is gitignored; run the script locally after cloning.

### Fetch the dataset

From the `rag_py` directory:

```bash
python scripts/fetch_react_docs.py
```

The script uses only the Python standard library. It prints progress for each URL and a summary when finished. Re-run it anytime to refresh the dataset.
