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
            each chunk stays within a logical section. The sectioner works on the raw
            document text so blank lines, list formatting, and hard breaks are preserved
            exactly (a reflowing splitter would collapse paragraph breaks and defeat the
            block decomposition in Pass 2).
  Pass 2 — Block-aware pack: decompose each section into atomic blocks (prose paragraphs,
            whole fenced code blocks, and whole MDX wrappers such as <Sandpack>) and
            greedily pack them up to a soft size target. A fenced code block or MDX
            wrapper is never severed mid-block; if a single block exceeds the target it
            becomes its own (slightly oversized) chunk rather than producing an unbalanced
            code fence. Overlap between consecutive chunks is carried as trailing prose
            only, so code is never duplicated as a meaningless stub at a boundary.
"""

import re

from langchain_core.documents import Document


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Markdown heading levels that act as natural section boundaries.
# The sectioner uses these to divide a document into semantically coherent sections
# before any block-based packing occurs. The key is the literal ATX prefix and the
# value is the metadata field that records the heading text for that level.
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Soft target for the number of characters per chunk. Tune this based on the token
# limit of your embedding model. Most models support ~512–8192 tokens; 1500 characters
# (~375 tokens) is a comfortable default that keeps code examples intact with their
# surrounding prose. This is a *soft* cap: an atomic block (a fenced code block or an
# MDX wrapper) is never broken to honor it, so a single large block may overflow.
CHUNK_SIZE = 1500

# Maximum number of characters of trailing prose carried from one chunk into the next.
# Overlap preserves continuity across a boundary without duplicating code: only prose
# is carried, and only when the previous chunk ended on a prose paragraph.
CHUNK_OVERLAP = 150

# Chunks smaller than this are merged into a neighbor so the pipeline never emits a
# near-empty stub (e.g. a lone heading) that would be a poor retrieval unit.
MIN_CHUNK_SIZE = 200

# MDX wrapper tags whose entire <Tag>...</Tag> span is treated as a single atomic block,
# including any code fences nested inside it. Two kinds of tags belong here:
#   - Example wrappers (<Sandpack>, <DiagramGroup>) that contain nested ```js / ```css
#     blocks which must stay together.
#   - Prose callouts (<Intro>, <DeepDive>, <Pitfall>, <ConsoleBlock>, <Note>,
#     <Deprecated>) that form a single self-contained aside; keeping them whole avoids
#     stranding their opening/closing tag lines in separate chunks and keeps the callout
#     interpretable on its own.
# Pure layout containers (<FullWidth>, <CodeDiagram>, <Recipes>) are intentionally NOT
# listed: their inner blocks (lists, images, nested <Sandpack>) should pack normally.
ATOMIC_MDX_TAGS = (
    "Sandpack",
    "DiagramGroup",
    "Intro",
    "DeepDive",
    "Pitfall",
    "ConsoleBlock",
    "Note",
    "Deprecated",
)

# Block kinds produced by _split_into_blocks.
_PROSE = "prose"
_CODE = "code"
_MDX = "mdx"

# Sentence boundary used when an oversized prose paragraph must be subdivided and when
# carrying trailing-prose overlap. It matches whitespace that follows ., ! or ? but the
# negative lookbehind (?<![0-9]\.) prevents splitting on the period of an ordered-list
# marker (e.g. "1." / "2."), which would otherwise strand a bare list number at a chunk
# boundary or flatten an enumerated list into run-on prose.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?<![0-9]\.)\s+")

# A prose fragment that is nothing but a bare ordered-list marker (e.g. "1."), used to
# avoid seeding a continuation chunk with a dangling list number.
_BARE_LIST_MARKER = re.compile(r"^\s*\d+\.\s*$")


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2 helpers: block decomposition and packing
# ──────────────────────────────────────────────────────────────────────────────

def _mdx_open_tag(stripped_line: str) -> str | None:
    """Return the atomic MDX tag name if the line opens one, else None."""
    for tag in ATOMIC_MDX_TAGS:
        if stripped_line == f"<{tag}>" or stripped_line.startswith(f"<{tag} "):
            return tag
    return None


def _split_long_prose(text: str, target: int) -> list[str]:
    """Split an oversized prose paragraph into sub-paragraphs on sentence boundaries.

    This is the only fallback that ever subdivides content by characters, and it is
    applied exclusively to prose — code blocks and MDX wrappers are never passed here.
    """
    if len(text) <= target:
        return [text]

    sentences = _SENTENCE_BOUNDARY.split(text)
    blocks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > target:
            blocks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        blocks.append(current)
    return blocks


def _split_into_blocks(text: str, max_prose: int = CHUNK_SIZE) -> list[tuple[str, str]]:
    """Decompose a section into ordered atomic blocks of (kind, content).

    Blocks are one of:
      - _CODE: a fenced code block, captured whole from its opening ``` to its closing ```.
      - _MDX:  an atomic MDX wrapper (e.g. <Sandpack>...</Sandpack>), captured whole
               including any nested code fences.
      - _PROSE: a run of non-blank, non-code, non-MDX lines (a paragraph). Oversized
                paragraphs are further split on sentence boundaries via _split_long_prose.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ── Fenced code block ────────────────────────────────────────────────
        if stripped.startswith("```"):
            buffer = [line]
            i += 1
            while i < n:
                buffer.append(lines[i])
                closed = lines[i].strip().startswith("```")
                i += 1
                if closed:
                    break
            blocks.append((_CODE, "\n".join(buffer)))
            continue

        # ── Atomic MDX wrapper (e.g. <Sandpack>) ─────────────────────────────
        tag = _mdx_open_tag(stripped)
        if tag is not None:
            close = f"</{tag}>"
            buffer = [line]
            depth = 1  # track nested same-tag wrappers so the right close is matched
            i += 1
            while i < n:
                current_line = lines[i]
                buffer.append(current_line)
                inner = current_line.strip()
                i += 1
                if _mdx_open_tag(inner) == tag:
                    depth += 1
                elif inner == close:
                    depth -= 1
                    if depth == 0:
                        break
            # If the wrapper is never closed, the buffer simply runs to the end of the
            # section rather than swallowing the rest of the document.
            blocks.append((_MDX, "\n".join(buffer)))
            continue

        # ── Blank line: paragraph separator ──────────────────────────────────
        if stripped == "":
            i += 1
            continue

        # ── Prose paragraph: until blank line or start of a code/MDX block ────
        buffer = [line]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if nxt == "" or nxt.startswith("```") or _mdx_open_tag(nxt) is not None:
                break
            buffer.append(lines[i])
            i += 1
        paragraph = "\n".join(buffer)
        for sub in _split_long_prose(paragraph, max_prose):
            blocks.append((_PROSE, sub))

    return _glue_lead_ins(blocks)


def _glue_lead_ins(blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Attach a colon-terminated prose lead-in to the code/MDX block it introduces.

    Prose like "Have a look at the result:" or "In this example, it is `MyApp`:" exists
    only to introduce the example that follows it. Merging the two into one atomic block
    keeps the introduction with its example so a chunk boundary can never strand the
    lead-in at the end of one chunk while its referent lands in the next.
    """
    glued: list[tuple[str, str]] = []
    i = 0
    n = len(blocks)
    while i < n:
        kind, text = blocks[i]
        is_lead_in = kind == _PROSE and text.rstrip().endswith(":")
        introduces_block = i + 1 < n and blocks[i + 1][0] in (_CODE, _MDX)
        if is_lead_in and introduces_block:
            next_kind, next_text = blocks[i + 1]
            glued.append((next_kind, f"{text}\n\n{next_text}"))
            i += 2
        else:
            glued.append((kind, text))
            i += 1
    return glued


def _trailing_prose_overlap(blocks: list[tuple[str, str]], overlap: int) -> str:
    """Return trailing prose to seed the next chunk, or "" if the chunk ended on code.

    Only the last block is considered: if it is prose, the last complete sentence(s) up
    to roughly `overlap` characters are carried; if it is code or an MDX wrapper, no
    overlap is carried so code is never duplicated as a stub. Carrying whole sentences
    (rather than a character slice) guarantees the next chunk never begins mid-sentence.
    """
    if not blocks or overlap <= 0:
        return ""
    kind, text = blocks[-1]
    if kind != _PROSE:
        return ""

    sentences = _SENTENCE_BOUNDARY.split(text)
    # Drop a trailing bare list marker (e.g. "1.") so the seed never starts on a dangling
    # list number once we reverse into it.
    while sentences and _BARE_LIST_MARKER.match(sentences[-1]):
        sentences.pop()
    if not sentences:
        return ""

    joined = " ".join(sentences)
    if len(joined) <= overlap:
        return joined

    collected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        addition = len(sentence) + (1 if collected else 0)
        if collected and total + addition > overlap:
            break
        collected.insert(0, sentence)
        total += addition

    # Always carry at least the final sentence whole, even if it exceeds `overlap`,
    # so the seed starts on a clean sentence boundary rather than a fragment.
    if not collected:
        collected = [sentences[-1]]
    return " ".join(collected)


def _pack_blocks(
    blocks: list[tuple[str, str]],
    target: int,
    overlap: int,
    heading: str | None = None,
) -> list[str]:
    """Greedily pack atomic blocks into chunk strings up to a soft `target` size.

    Blocks are joined with blank lines. A new chunk is started whenever appending the
    next block would exceed `target` (unless the current chunk is empty, so a single
    oversized block still becomes one whole chunk). Each new chunk after the first is
    seeded with the section heading (for standalone interpretability) and any trailing
    prose overlap from the chunk just flushed.
    """
    chunks: list[str] = []
    current: list[tuple[str, str]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(text for _, text in current))
            current = []
            current_len = 0

    for kind, text in blocks:
        addition = len(text) + (2 if current else 0)
        if current and current_len + addition > target:
            flushed = current
            flush()

            seed: list[tuple[str, str]] = []
            if heading and not text.lstrip().startswith("#"):
                seed.append((_PROSE, heading))
            overlap_text = _trailing_prose_overlap(flushed, overlap)
            if overlap_text and overlap_text.strip() != (heading or "").strip():
                seed.append((_PROSE, overlap_text))
            for seed_block in seed:
                current.append(seed_block)
                current_len += len(seed_block[1]) + (2 if len(current) > 1 else 0)

        current.append((kind, text))
        current_len += len(text) + (2 if len(current) > 1 else 0)

    flush()
    return chunks


def _merge_tiny_chunks(documents: list[Document], min_size: int) -> list[Document]:
    """Merge any chunk smaller than `min_size` into an adjacent chunk.

    A tiny chunk is folded into its previous neighbor (inheriting that neighbor's
    metadata); a tiny leading chunk is folded forward into the following chunk. This
    removes near-empty stubs such as a lone heading with a single link.
    """
    if not documents:
        return documents

    merged: list[Document] = []
    for doc in documents:
        if merged and len(doc.page_content) < min_size:
            previous = merged[-1]
            previous.page_content = f"{previous.page_content}\n\n{doc.page_content}"
        else:
            merged.append(doc)

    if len(merged) >= 2 and len(merged[0].page_content) < min_size:
        following = merged[1]
        following.page_content = f"{merged[0].page_content}\n\n{following.page_content}"
        merged.pop(0)

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1 helper: structure-aware sectioning
# ──────────────────────────────────────────────────────────────────────────────

# Matches an ATX heading line, capturing the leading #s and the heading text.
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Matches a trailing MDX heading anchor such as " {/*step-1-foo*/}" so it can be dropped
# from heading metadata while remaining in the chunk body.
_HEADING_ANCHOR = re.compile(r"\s*\{/\*.*?\*/\}\s*$")


def _split_into_sections(text: str) -> list[Document]:
    """Divide raw Markdown into sections at configured ATX headings, preserving formatting.

    Unlike a reflowing splitter, this keeps the original blank lines, list indentation,
    and hard line breaks exactly as written so Pass 2 can see real paragraph boundaries.
    Headings inside fenced code blocks are ignored. Each section carries the running
    heading path as metadata (e.g. {"Header 1": ..., "Header 2": ...}); a heading resets
    its own level and any deeper ones while shallower ancestors are retained. The trailing
    {/*anchor*/} is stripped from metadata but left intact in the chunk body. Content that
    precedes the first heading becomes a leading section with no heading metadata.
    """
    level_field = {len(prefix): field for prefix, field in HEADERS_TO_SPLIT_ON}

    sections: list[Document] = []
    current_lines: list[str] = []
    current_meta: dict[str, str] = {}
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in current_lines):
            content = "\n".join(current_lines).strip("\n")
            sections.append(Document(page_content=content, metadata=dict(current_meta)))

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        match = None if in_fence else _ATX_HEADING.match(stripped)
        level = len(match.group(1)) if match else 0
        if match and level in level_field:
            flush()
            for lvl, field in level_field.items():
                if lvl >= level:
                    current_meta.pop(field, None)
            current_meta[level_field[level]] = _HEADING_ANCHOR.sub("", match.group(2)).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    return sections


# ──────────────────────────────────────────────────────────────────────────────
# Chunking logic
# ──────────────────────────────────────────────────────────────────────────────

def chunk_document(document: Document) -> list[Document]:
    """Split a single Document into smaller chunks suitable for embedding and retrieval.

    The function applies two passes in sequence:

    1. Structure-aware sectioner (_split_into_sections) — respects the document's heading
       structure so related content stays together, working on the raw text so blank
       lines and list formatting are preserved. Each section inherits heading text as
       metadata (e.g. {"Header 2": "Installation"}) which can later be used to filter or
       display provenance alongside retrieved results.

    2. Block-aware packer — decomposes each section into atomic blocks (prose, fenced
       code, MDX wrappers) and greedily packs them up to CHUNK_SIZE without ever severing
       a code block or MDX wrapper. Tiny chunks are then merged into a neighbor.

    Args:
        document: A LangChain Document object containing the raw page content and any
                  source metadata (e.g. URL, filename) attached during loading.

    Returns:
        A list of Document chunks ready to be embedded and upserted into the vector store.
    """

    # ── Pass 1: structure-aware split ─────────────────────────────────────────
    # Divide the document along its Markdown heading hierarchy while preserving the
    # raw formatting (blank lines, list indentation, hard breaks). The heading text is
    # kept inside each chunk so the LLM receives full context about which section a
    # chunk belongs to.
    sections = _split_into_sections(document.page_content)

    # ── Pass 2: block-aware pack ───────────────────────────────────────────────
    # Decompose each section into atomic blocks and pack them up to CHUNK_SIZE,
    # carrying prose-only overlap and the section heading into continuation chunks.
    chunks: list[Document] = []
    for section in sections:
        # Propagate the original document's metadata (e.g. source URL) into every
        # section. The heading metadata added by the sectioner is merged on top so
        # both are available at retrieval time.
        metadata = {**document.metadata, **section.metadata}

        first_line = section.page_content.lstrip().split("\n", 1)[0]
        heading = first_line if first_line.startswith("#") else None

        blocks = _split_into_blocks(section.page_content, max_prose=CHUNK_SIZE)
        for chunk_text in _pack_blocks(blocks, CHUNK_SIZE, CHUNK_OVERLAP, heading):
            chunks.append(Document(page_content=chunk_text, metadata=dict(metadata)))

    return _merge_tiny_chunks(chunks, MIN_CHUNK_SIZE)


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. Please import it as a module.")
