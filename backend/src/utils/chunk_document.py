"""
Chunking is one of the most important steps in building a RAG pipeline. Before documents
can be stored in a vector database and retrieved later, they need to be split into smaller
pieces called "chunks". This matters because:

  1. Embedding models have a token limit — you can't embed an entire document at once.
  2. Smaller, focused chunks produce more precise embeddings, which leads to better
     semantic search results at retrieval time.
  3. Returning a focused chunk as context to the LLM is more useful than returning an
     entire document that may contain irrelevant content.

This module implements a two-pass chunking strategy tailored for Markdown documents:
  Pass 1 — Structure-aware split: divide the document along its heading hierarchy so
            each chunk stays within a logical section.
  Pass 2 — Size-aware split: further subdivide any sections that are still too large
            for the embedding model, while keeping some overlap so context isn't lost
            at chunk boundaries.
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Markdown heading levels that act as natural section boundaries.
# The splitter uses these to divide a document into semantically coherent sections
# before any character-based splitting occurs.
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Maximum number of characters per chunk. Tune this based on the token limit of
# your embedding model. Most models support ~512–8192 tokens; 1000 characters is
# a conservative default that works well across model families.
CHUNK_SIZE = 1000

# Number of characters shared between consecutive chunks. Overlap prevents a
# sentence or concept from being split across two chunks in a way that would make
# either chunk meaningless when retrieved in isolation.
CHUNK_OVERLAP = 150


# ──────────────────────────────────────────────────────────────────────────────
# Chunking logic
# ──────────────────────────────────────────────────────────────────────────────

def chunk_document(document: Document) -> list[Document]:
    """Split a single Document into smaller chunks suitable for embedding and retrieval.

    The function applies two splitters in sequence:

    1. MarkdownHeaderTextSplitter — respects the document's heading structure so
       related content stays together. Each resulting section inherits heading text
       as metadata (e.g. {"Header 2": "Installation"}) which can later be used to
       filter or display provenance alongside retrieved results.

    2. RecursiveCharacterTextSplitter — further splits any section that still exceeds
       CHUNK_SIZE. The MARKDOWN language preset defines a priority-ordered list of
       split points (headers → blank lines → sentences → characters) so the splitter
       always tries to break on the most natural boundary first.

    Args:
        document: A LangChain Document object containing the raw page content and any
                  source metadata (e.g. URL, filename) attached during loading.

    Returns:
        A list of Document chunks ready to be embedded and upserted into the vector store.
    """

    # ── Pass 1: structure-aware split ─────────────────────────────────────────
    # Divide the document along its Markdown heading hierarchy. Setting
    # strip_headers=False keeps heading text inside each chunk so the LLM
    # receives full context about which section a chunk belongs to.
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    sections = md_splitter.split_text(document.page_content)

    # Propagate the original document's metadata (e.g. source URL) into every
    # section. The heading metadata added by the splitter is merged on top so
    # both are available at retrieval time.
    for section in sections:
        section.metadata = {**document.metadata, **section.metadata}

    # ── Pass 2: size-aware split ───────────────────────────────────────────────
    # Sections that exceed CHUNK_SIZE are further divided. Using from_language
    # with Language.MARKDOWN sets a Markdown-aware split priority, meaning the
    # splitter will try to break on headings, then paragraphs, then sentences,
    # before falling back to raw character boundaries.
    recursive_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return recursive_splitter.split_documents(sections)


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
