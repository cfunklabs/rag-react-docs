from pathlib import Path
from langchain_core.documents import Document
from dotenv import load_dotenv
from .config import load_colors


load_dotenv()


_colors = load_colors()
RED = _colors["RED"]
RESET = _colors["RESET"]


def load_document_at_path(file_path: str) -> Document | None:
  try:
    text = Path(file_path).read_text(encoding="utf-8")
    document = Document(page_content=text, metadata={"source": file_path})
    return document
  except Exception as e:
    print(f"{RED}Error loading document {file_path}: {e}{RESET}")
    return None

if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
