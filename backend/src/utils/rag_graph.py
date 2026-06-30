"""LangGraph definition for the retrieval + generation half of the RAG pipeline.

The graph is intentionally small and linear:

    START -> retrieve -> generate -> END

`retrieve` pulls the top-k chunks for the question from ChromaDB and `generate` feeds those
chunks to Claude as grounding context. Keeping each step as a discrete node (rather than a
single LCEL chain) makes the data flow inspectable and leaves room to add nodes later (e.g.
query rewriting or re-ranking) without restructuring the call sites.
"""

from pathlib import Path
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

from .config import load_rag_query_config
from .retrieve_chunks import retrieve_chunks


_DOCS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "docs"

_query_config = load_rag_query_config()
GENERATION_MODEL = _query_config["generation_model"]
DEFAULT_TOP_K = _query_config["top_k"]

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the React documentation. "
    "Answer the user's question using ONLY the information in the provided context. "
    "The context is a set of excerpts retrieved from the React docs, each labeled with its "
    "source file and section heading path. If the context does not contain enough information "
    "to answer, say so plainly instead of guessing. When helpful, reference the relevant "
    "section headings so the user knows where the information comes from. Prefer concrete code "
    "examples from the context when they are present."
)


class RAGState(TypedDict):
    """State threaded through the graph nodes."""

    question: str
    k: int
    context: list[Document]
    answer: str


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


def _format_context(chunks: list[Document]) -> str:
    """Render retrieved chunks into a single grounding string for the prompt."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks):
        label = format_source_label(chunk.metadata)
        blocks.append(f"[Source {index + 1}] {label}\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


def _retrieve(state: RAGState) -> dict:
    k = state.get("k") or DEFAULT_TOP_K
    chunks = retrieve_chunks(state["question"], k)
    return {"context": chunks}


def _generate(state: RAGState) -> dict:
    chunks = state["context"]
    if not chunks:
        return {
            "answer": "I could not find anything relevant in the indexed documentation to answer that."
        }

    context = _format_context(chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n\n{context}\n\nQuestion: {state['question']}"),
    ]
    llm = ChatAnthropic(model=GENERATION_MODEL, temperature=0.0)
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_rag_graph():
    """Compile and return the retrieve -> generate RAG graph."""
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("generate", _generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
