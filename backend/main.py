from pathlib import Path
import chromadb
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from src.utils.config import load_colors, load_pyproject


load_dotenv()


_colors = load_colors()
RED = _colors["RED"]
GREEN = _colors["GREEN"]
YELLOW = _colors["YELLOW"]
RESET = _colors["RESET"]

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOC_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
DATASTORE_DIR = Path(__file__).resolve().parent / "rag_datastore"
RAG_COLLECTION_NAME = load_pyproject()["tool"]["rag_db"]["rag_doc_collection_name"]


def llm_health_check():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
    response = llm.invoke("Say this exact phrase, no more and no less: 'Claude API interface is up and running'")
    return True if response.content.lower() == "claude api interface is up and running" else False


def docs_check():
    if not DOCS_DIR.is_dir():
        print(f" {RED}FAILED{RESET}")
        print(f"{RED}Docs directory not found at {DOCS_DIR}{RESET}")
        return False
    file_count = sum(
        1 for p in DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in DOC_EXTENSIONS
    )
    if file_count == 0:
        print(f" {RED}FAILED{RESET}")
        print(f"{RED}No document files found in {DOCS_DIR}{RESET}")
        return False
    print(f" {GREEN}PASSED{RESET} ({file_count} document{'s' if file_count != 1 else ''} found)")
    return True


def datastore_check():
    if not DATASTORE_DIR.is_dir():
        print(f" {RED}FAILED{RESET}")
        print(f"{RED}RAG datastore not found at {DATASTORE_DIR}. Run init_db first.{RESET}")
        return False
    try:
        client = chromadb.PersistentClient(path=str(DATASTORE_DIR))
        collection = client.get_collection(name=RAG_COLLECTION_NAME)
        doc_count = collection.count()
    except Exception as e:
        print(f" {RED}FAILED{RESET}")
        print(f"{RED}Collection '{RAG_COLLECTION_NAME}' not initialized: {e}. Run init_db first.{RESET}")
        return False
    doc_label = f"{doc_count} document{'s' if doc_count != 1 else ''}"
    if doc_count == 0:
        doc_label = f"{YELLOW}{doc_label}{RESET}"
    print(f" {GREEN}PASSED{RESET} (collection '{RAG_COLLECTION_NAME}', {doc_label})")
    return True


def startup_check():
    print("Checking docs directory...", end="")
    if not docs_check():
        return False

    print("Checking RAG datastore...", end="")
    if not datastore_check():
        return False

    print("Checking LLM service health...", end="")
    is_llm_healthy = llm_health_check()
    print(f" {GREEN if is_llm_healthy else RED}{'PASSED' if is_llm_healthy else 'FAILED'}{RESET}")
    if not is_llm_healthy:
        return False

    return True


def get_md_doc_file_paths() -> list[Path]:
    md_extensions = {".md", ".markdown"}
    return [p for p in DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in md_extensions]


def process_documents():
    # Get a list of all of the .md files in the docs directory

    # Iterate over the list of .md files and process each one
    for file_path in md_file_paths:
        # Step 1: Load the document at the file path

        # Step 2: Chunk the document

        # Step 3: Compute the embeddings for the document

        # Step 4: Upsert the document into the vector database


def main():
    if not startup_check():
        print(f"{RED}Startup check failed. Exiting...{RESET}")
        return

if __name__ == "__main__":
    main()
