import argparse
import contextlib
import io

from dotenv import load_dotenv

from main import datastore_check, llm_health_check
from src.utils.config import load_colors, load_rag_query_config
from src.utils.get_rag_collection import get_rag_collection
from src.utils.rag_graph import DEFAULT_TOP_K, build_rag_graph, format_source_label


load_dotenv()


_colors = load_colors()
RED = _colors["RED"]
GREEN = _colors["GREEN"]
YELLOW = _colors["YELLOW"]
RESET = _colors["RESET"]


def _collection_has_documents() -> bool:
    try:
        return get_rag_collection().count() > 0
    except Exception:
        return False


def preflight_check(verbose: bool) -> bool:
    """Verify the datastore and LLM are ready before querying.

    The step-by-step "Checking..." diagnostics are only shown when `verbose` is set; the
    routine output is otherwise suppressed so a normal run prints just the answer and its
    sources. Actionable problems (empty collection, failed checks) are always reported.
    """

    def quiet():
        return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())

    with quiet():
        print("Checking RAG datastore...", end="")
        datastore_ok = datastore_check()
    if not datastore_ok:
        print(f"{RED}RAG datastore check failed. Run 'uv run main.py' after initializing the DB.{RESET}")
        return False

    if not _collection_has_documents():
        print(
            f"{YELLOW}Warning: the collection is empty. Run 'uv run main.py' to ingest documents "
            f"before querying.{RESET}"
        )
        return False

    with quiet():
        print("Checking LLM service health...", end="")
        is_llm_healthy = llm_health_check()
        print(f" {GREEN if is_llm_healthy else RED}{'PASSED' if is_llm_healthy else 'FAILED'}{RESET}")
    if not is_llm_healthy:
        print(f"{RED}LLM health check failed. Confirm ANTHROPIC_API_KEY is set correctly in .env.{RESET}")
        return False

    return True


def _extract_text(content) -> str:
    """Normalize a message chunk's content to plain text.

    Anthropic streaming deltas are usually plain strings, but content can also arrive as a
    list of content blocks; in that case we concatenate the text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def print_sources(context: list, show_scores: bool) -> None:
    if not context:
        return
    print(f"\n\n{YELLOW}Sources:{RESET}")
    seen: set[str] = set()
    for chunk in context:
        label = format_source_label(chunk.metadata)
        if label in seen:
            continue
        seen.add(label)
        if show_scores:
            distance = chunk.metadata.get("distance")
            score = f" {YELLOW}(distance: {distance:.4f}){RESET}" if distance is not None else ""
            print(f"  - {label}{score}")
        else:
            print(f"  - {label}")


def run_streaming(graph, state: dict, show_scores: bool) -> None:
    final_state: dict = {}
    streamed_any = False

    for mode, chunk in graph.stream(state, stream_mode=["messages", "values"]):
        if mode == "messages":
            message_chunk, metadata = chunk
            if metadata.get("langgraph_node") != "generate":
                continue
            text = _extract_text(message_chunk.content)
            if text:
                print(text, end="", flush=True)
                streamed_any = True
        elif mode == "values":
            final_state = chunk

    # The empty-context fallback returns an answer without invoking the LLM, so nothing
    # streams; surface that answer here instead.
    if not streamed_any and final_state.get("answer"):
        print(final_state["answer"], end="")

    print_sources(final_state.get("context", []), show_scores)
    print()


def run_blocking(graph, state: dict, show_scores: bool) -> None:
    final_state = graph.invoke(state)
    print(final_state.get("answer", ""), end="")
    print_sources(final_state.get("context", []), show_scores)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the LangChain RAG demo")
    parser.add_argument("question", type=str, help="The question to ask about the React docs.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K} from pyproject.toml).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full answer instead of streaming tokens as they arrive.",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show the retrieval distance for each cited source.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show diagnostic output (preflight datastore and LLM health checks).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not preflight_check(args.verbose):
        return

    k = args.k if args.k is not None else load_rag_query_config()["top_k"]
    graph = build_rag_graph()
    state = {"question": args.question, "k": k}

    print()
    if args.no_stream:
        run_blocking(graph, state, args.show_scores)
    else:
        run_streaming(graph, state, args.show_scores)


if __name__ == "__main__":
    main()
