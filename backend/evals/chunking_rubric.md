# Chunking Quality Rubric

A qualitative evaluation rubric for the two-pass Markdown chunker in
[`backend/src/utils/chunk_document.py`](../src/utils/chunk_document.py). An AI agent reads a
source document together with its ordered chunks and produces a reproducible,
criterion-based assessment, recorded in
[`chunking_assessment_template.json`](./chunking_assessment_template.json).

## What the chunker does

Understanding the algorithm is necessary to score it fairly:

- **Pass 1 — structure-aware split.** `MarkdownHeaderTextSplitter` splits on `#`, `##`, and
  `###`. `strip_headers=False`, so heading text stays inside each chunk. The original
  document metadata (e.g. `source`) is merged with heading metadata
  (`Header 1` / `Header 2` / `Header 3`).
- **Pass 2 — size-aware split.** `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)`
  subdivides any section still larger than `CHUNK_SIZE`. It prefers natural Markdown
  boundaries (headings, then blank lines, then sentences, then characters).
- **Configuration.** `CHUNK_SIZE = 1000` characters, `CHUNK_OVERLAP = 150` characters.

## Source material characteristics

The corpus is React documentation (see
[`backend/src/utils/fetch_react_docs.py`](../src/utils/fetch_react_docs.py)). Evaluators
should expect:

- Heavy use of fenced code blocks and JSX examples.
- YAML frontmatter at the top of each file (delimited by `---`).
- Tables, nested lists, and callouts.
- Deeply nested heading hierarchies.

## How to score

- Each of the 10 criteria below receives an integer score from **1 to 5**.
- Every score requires a **short justification** that cites specific chunk indices as
  evidence.
- Separately, evaluate the four binary **auto-fail flags**. Any flag that is `true` caps
  the document's `overall_rating` at **Needs Work**, regardless of the weighted average.
- The `weighted_score` is the weighted average of the 10 criteria (weights below).

### Scoring scale (applies to every criterion)

| Score | Label      | Meaning                                                        |
| ----- | ---------- | -------------------------------------------------------------- |
| 5     | Excellent  | Fully satisfies the criterion; no observable issues.           |
| 4     | Good       | Minor, non-impactful issues.                                   |
| 3     | Acceptable | Noticeable issues that mildly degrade retrieval quality.       |
| 2     | Poor       | Frequent issues that meaningfully degrade quality.             |
| 1     | Failing    | Criterion largely unmet.                                       |

## Criteria

### 1. Semantic Coherence (weight 2)

Each chunk covers one coherent topic or concept; no jarring topic switches mid-chunk.

- **5** — Every chunk is a self-contained, single-topic unit.
- **3** — A few chunks bundle loosely related topics or switch subject midway.
- **1** — Chunks routinely mix unrelated topics, making them hard to use as retrieval units.

### 2. Structural Boundary Respect (weight 2)

Splits align with the Markdown heading hierarchy; heading context is retained in-chunk
(consistent with `strip_headers=False`); sections are not split at arbitrary points when a
heading boundary was available.

- **5** — Chunk boundaries consistently fall on heading boundaries; headings are preserved.
- **3** — Some chunks split away from available heading boundaries or drop heading context.
- **1** — Boundaries ignore the heading structure; sections are fragmented arbitrarily.

### 3. Code Block Integrity (weight 2)

Fenced code blocks and JSX examples are not severed mid-block; opening and closing fences
stay together; a code sample stays with the prose that introduces it where feasible.
(Weighted highly because the corpus is code-dense.)

- **5** — All code blocks remain intact and stay near their explanatory prose.
- **3** — Occasional long code blocks are split, but fences remain balanced.
- **1** — Code blocks are frequently severed, leaving unbalanced or meaningless fragments.

### 4. Sentence & Paragraph Integrity (weight 1)

No breaks mid-sentence or mid-word; paragraphs and list items are kept intact at boundaries.

- **5** — Boundaries fall cleanly between sentences and paragraphs.
- **3** — A few chunks break mid-paragraph but remain readable.
- **1** — Chunks frequently break mid-sentence or mid-word.

### 5. Chunk Size Appropriateness (weight 1)

Chunks respect `CHUNK_SIZE = 1000` without large overflow, and avoid tiny low-value
fragments (e.g. a lone heading or a stub line).

- **5** — Chunk sizes are well distributed; no large overflow and no trivial fragments.
- **3** — A few chunks are noticeably oversized or trivially small.
- **1** — Many chunks badly overflow the limit or are near-empty stubs.

### 6. Overlap Quality (weight 1)

The `CHUNK_OVERLAP = 150` produces meaningful continuity at boundaries without excessive or
noisy duplication.

- **5** — Overlap preserves continuity across boundaries with minimal redundancy.
- **3** — Overlap is inconsistent — sometimes missing context, sometimes overly redundant.
- **1** — Overlap is absent where needed or floods chunks with duplicated noise.

### 7. Standalone Interpretability (weight 2)

A chunk read in isolation (as it would be at retrieval time) is understandable, with enough
heading or contextual cues to know what it describes.

- **5** — Every chunk makes sense on its own without the surrounding document.
- **3** — Some chunks need neighboring context to be interpretable.
- **1** — Many chunks are unintelligible in isolation (dangling pronouns, orphaned code).

### 8. Content Completeness & No Loss (weight 1)

The union of chunks reconstructs the source content; nothing is dropped, and there is no
unexpected duplication beyond the intended overlap. Frontmatter is handled sensibly.

- **5** — All source content is represented exactly once (modulo intended overlap).
- **3** — Minor content is dropped or duplicated beyond the overlap window.
- **1** — Significant content is missing or heavily duplicated.

### 9. Metadata Accuracy (weight 1)

`source` plus `Header 1` / `Header 2` / `Header 3` metadata are present and correctly
reflect each chunk's location in the document.

- **5** — Metadata is complete and accurately reflects each chunk's section.
- **3** — Metadata is present but occasionally stale or imprecise.
- **1** — Metadata is missing or wrong for many chunks.

### 10. Retrieval Utility (weight 2)

Each chunk maps cleanly to plausible user questions about React; chunks are neither too
broad nor too fragmented to serve as effective retrieval units.

- **5** — Chunks are sized and scoped to directly answer realistic React questions.
- **3** — Chunks are usable but often too broad or too granular for clean retrieval.
- **1** — Chunks would rarely surface usefully for realistic queries.

## Critical auto-fail flags (binary)

Evaluate these independently of the 1-5 scores. If any flag is `true`, the document's
`overall_rating` is capped at **Needs Work** regardless of the weighted average.

| Flag                | Trigger                                                                        |
| ------------------- | ------------------------------------------------------------------------------ |
| `broken_code_fence` | A code block is split so a chunk contains an unbalanced fence.                 |
| `mid_sentence_split`| A chunk starts or ends mid-sentence or mid-word with no overlap recovery.      |
| `content_loss`      | Source content is missing from all chunks.                                     |
| `metadata_missing`  | `source` or expected heading metadata is absent or incorrect.                  |

## Overall rating

1. Compute `weighted_score` = weighted average of the 10 criteria using the weights above
   (total weight = 16).
2. Map the weighted score to a rating:

| Weighted score | Overall rating |
| -------------- | -------------- |
| 4.5 – 5.0      | Excellent      |
| 3.5 – 4.49     | Good           |
| 2.5 – 3.49     | Acceptable     |
| 1.0 – 2.49     | Needs Work     |

3. If any auto-fail flag is `true`, force `overall_rating` to **Needs Work** even if the
   weighted score is higher.

## Output

Record one populated copy of
[`chunking_assessment_template.json`](./chunking_assessment_template.json) per evaluated
document. Use the `summary` and `recommendations` fields to capture qualitative findings and
concrete tuning suggestions (e.g. adjusting `CHUNK_SIZE`, `CHUNK_OVERLAP`, or
`HEADERS_TO_SPLIT_ON`).
