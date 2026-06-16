from pathlib import Path
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.utils.config import load_colors


load_dotenv()


_colors = load_colors()
RED = _colors["RED"]
YELLOW = _colors["YELLOW"]
RESET = _colors["RESET"]


def load_document_at_path(file_path: str):
  try:
    text = Path(file_path).read_text(encoding="utf-8")
    document = Document(page_content=text, metadata={"source": file_path})
    return document
  except Exception as e:
    print(f"Error loading document {file_path}: {e}")
    return None

if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
