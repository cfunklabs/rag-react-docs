import chromadb
import tomllib
from pathlib import Path


def init_db():
  chroma_client = chromadb.PersistentClient(path="./rag_datastore")

  with open(Path("pyproject.toml"), "rb") as f:
    collection_name = tomllib.load(f)["tool"]["rag_db"]["rag_doc_collection_name"]
    
  collection = chroma_client.get_or_create_collection(name=collection_name)

  print(f"Collection {collection_name} created")
  print("DB initialized")
  return collection


def main() -> int:
  try:
    init_db()
  except Exception as e:
    print(f"Error initializing DB: {e}")
    return 1
  return 0


if __name__ == "__main__":
    raise SystemExit(main())
