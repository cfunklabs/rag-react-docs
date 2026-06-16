from pathlib import Path
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def load_document_at_path(file_path: str):
  try:
    text = Path(file_path).read_text(encoding="utf-8")
    document = Document(page_content=text, metadata={"source": file_path})

    print(f"Loaded document {file_path}")
    print("-" * 100)
    print(document.page_content)
    print("-" * 100)
    print(document.metadata)
    print("-" * 100)
    print("\n")

  except Exception as e:
    print(f"Error loading document {file_path}: {e}")
    return []

  return document

