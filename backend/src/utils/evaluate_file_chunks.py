import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.utils.config import load_colors
from src.utils.load_document_at_path import load_document_at_path


_colors = load_colors()
RED = _colors["RED"]
GREEN = _colors["GREEN"]
YELLOW = _colors["YELLOW"]
RESET = _colors["RESET"]

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = _BACKEND_DIR.parent / "docs"
EVALS_DIR = _BACKEND_DIR / "evals"
RESULTS_DIR = EVALS_DIR / "results"
RUBRIC_PATH = EVALS_DIR / "chunking_rubric.md"
JUDGE_MODEL = "claude-sonnet-4-6"

CRITERIA_WEIGHTS = {
    "semantic_coherence": 2,
    "structural_boundary_respect": 2,
    "code_block_integrity": 2,
    "sentence_paragraph_integrity": 1,
    "chunk_size_appropriateness": 1,
    "overlap_quality": 1,
    "standalone_interpretability": 2,
    "content_completeness_no_loss": 1,
    "metadata_accuracy": 1,
    "retrieval_utility": 2,
}
TOTAL_WEIGHT = sum(CRITERIA_WEIGHTS.values())
AUTO_FAIL_FLAGS = (
    "broken_code_fence",
    "mid_sentence_split",
    "content_loss",
    "metadata_missing",
)


class CriterionScore(BaseModel):
    """A single rubric criterion graded on the 1-5 scale."""

    score: int = Field(ge=1, le=5, description="Integer score from 1 (failing) to 5 (excellent).")
    justification: str = Field(description="Short rationale citing specific chunk indices as evidence.")
    evidence_chunk_indices: list[int] = Field(
        default_factory=list,
        description="Zero-based indices of the chunks that most support this score.",
    )


class AutoFailFlag(BaseModel):
    """A binary critical-defect flag. When triggered, the overall rating is capped at 'Needs Work'."""

    triggered: bool = Field(description="Whether this critical defect was observed.")
    details: str = Field(default="", description="Where/how the defect occurred; empty if not triggered.")


class ChunkingAssessment(BaseModel):
    """Structured judge output. Excludes weighted_score/overall_rating, which are computed in Python."""

    semantic_coherence: CriterionScore
    structural_boundary_respect: CriterionScore
    code_block_integrity: CriterionScore
    sentence_paragraph_integrity: CriterionScore
    chunk_size_appropriateness: CriterionScore
    overlap_quality: CriterionScore
    standalone_interpretability: CriterionScore
    content_completeness_no_loss: CriterionScore
    metadata_accuracy: CriterionScore
    retrieval_utility: CriterionScore

    broken_code_fence: AutoFailFlag
    mid_sentence_split: AutoFailFlag
    content_loss: AutoFailFlag
    metadata_missing: AutoFailFlag

    summary: str = Field(description="Qualitative summary of the chunking quality for this document.")
    recommendations: str = Field(
        description="Concrete tuning suggestions (e.g. CHUNK_SIZE, CHUNK_OVERLAP, HEADERS_TO_SPLIT_ON)."
    )


def _assessment_output_paths(source: str) -> tuple[Path, Path]:
    """Return (json_path, md_path) under RESULTS_DIR, mirroring the source's path under DOCS_DIR.

    Example: docs/Learn React/GET STARTED/Quick Start/Quick Start.md
          -> backend/evals/results/Learn React/GET STARTED/Quick Start/Quick Start_chunking_assessment.{json,md}
    """
    relative = Path(source).resolve().relative_to(DOCS_DIR.resolve())
    stem = f"{relative.stem}_chunking_assessment"
    destination_dir = RESULTS_DIR / relative.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    return destination_dir / f"{stem}.json", destination_dir / f"{stem}.md"


def _compute_weighted_score(criteria: dict) -> float:
    """Weighted average of the per-criterion scores, on the 1-5 scale."""
    total = sum(criteria[name]["score"] * weight for name, weight in CRITERIA_WEIGHTS.items())
    return round(total / TOTAL_WEIGHT, 2)


def _map_overall_rating(weighted_score: float, any_flag_triggered: bool) -> str:
    """Map a weighted score to a rating band; any auto-fail flag caps the rating at 'Needs Work'."""
    if any_flag_triggered:
        return "Needs Work"
    if weighted_score >= 4.5:
        return "Excellent"
    if weighted_score >= 3.5:
        return "Good"
    if weighted_score >= 2.5:
        return "Acceptable"
    return "Needs Work"


def _build_assessment_dict(
    assessment: ChunkingAssessment,
    source: str,
    source_length_chars: int,
    chunk_count: int,
) -> dict:
    """Assemble a dict matching backend/evals/chunking_assessment_template.json with computed totals."""
    criteria = {
        name: {
            "weight": weight,
            "score": getattr(assessment, name).score,
            "justification": getattr(assessment, name).justification,
            "evidence_chunk_indices": getattr(assessment, name).evidence_chunk_indices,
        }
        for name, weight in CRITERIA_WEIGHTS.items()
    }
    auto_fail_flags = {
        flag: {
            "triggered": getattr(assessment, flag).triggered,
            "details": getattr(assessment, flag).details,
        }
        for flag in AUTO_FAIL_FLAGS
    }
    weighted_score = _compute_weighted_score(criteria)
    any_flag_triggered = any(flag["triggered"] for flag in auto_fail_flags.values())
    overall_rating = _map_overall_rating(weighted_score, any_flag_triggered)

    return {
        "document": {
            "source": source,
            "source_length_chars": source_length_chars,
            "chunk_count": chunk_count,
        },
        "criteria": criteria,
        "auto_fail_flags": auto_fail_flags,
        "weighted_score": weighted_score,
        "overall_rating": overall_rating,
        "overall_rating_options": ["Excellent", "Good", "Acceptable", "Needs Work"],
        "summary": assessment.summary,
        "recommendations": assessment.recommendations,
    }


def _render_markdown_report(assessment: dict) -> str:
    """Render a human-readable Markdown report from an assessment dict."""
    document = assessment["document"]
    lines: list[str] = []
    lines.append("# Chunking Assessment")
    lines.append("")
    lines.append(f"- **Source:** `{document['source']}`")
    lines.append(f"- **Source length (chars):** {document['source_length_chars']}")
    lines.append(f"- **Chunk count:** {document['chunk_count']}")
    lines.append(f"- **Weighted score:** {assessment['weighted_score']} / 5")
    lines.append(f"- **Overall rating:** {assessment['overall_rating']}")
    lines.append("")

    lines.append("## Criteria")
    lines.append("")
    lines.append("| Criterion | Score | Weight | Evidence chunks |")
    lines.append("| --- | --- | --- | --- |")
    for name, data in assessment["criteria"].items():
        label = name.replace("_", " ").title()
        evidence = ", ".join(str(i) for i in data["evidence_chunk_indices"]) or "-"
        lines.append(f"| {label} | {data['score']}/5 | {data['weight']} | {evidence} |")
    lines.append("")
    for name, data in assessment["criteria"].items():
        label = name.replace("_", " ").title()
        lines.append(f"- **{label} ({data['score']}/5):** {data['justification']}")
    lines.append("")

    lines.append("## Critical auto-fail flags")
    lines.append("")
    for flag, data in assessment["auto_fail_flags"].items():
        label = flag.replace("_", " ").title()
        status = "TRIGGERED" if data["triggered"] else "ok"
        detail = f" — {data['details']}" if data["details"] else ""
        lines.append(f"- **{label}:** {status}{detail}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(assessment["summary"])
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(assessment["recommendations"])
    lines.append("")
    return "\n".join(lines)


def _build_judge_prompt(rubric_text: str, source_text: str, chunks: list[Document]) -> str:
    """Build the user-message portion of the judge prompt: rubric + source document + enumerated chunks."""
    chunk_blocks: list[str] = []
    for index, chunk in enumerate(chunks):
        metadata = json.dumps(chunk.metadata, ensure_ascii=False)
        chunk_blocks.append(
            f"--- CHUNK {index} START ---\n"
            f"metadata: {metadata}\n"
            f"content:\n{chunk.page_content}\n"
            f"--- CHUNK {index} END ---"
        )
    chunks_section = "\n\n".join(chunk_blocks)
    return (
        "# RUBRIC\n"
        "Apply this rubric exactly. Score every criterion and evaluate every auto-fail flag.\n\n"
        f"{rubric_text}\n\n"
        "# SOURCE DOCUMENT\n"
        "This is the original Markdown document the chunks were derived from.\n\n"
        f"{source_text}\n\n"
        "# CHUNKS\n"
        "These are the ordered chunks produced by the chunker, each with its index and metadata.\n\n"
        f"{chunks_section}\n\n"
        "# TASK\n"
        "- Score each of the 10 criteria from 1 to 5, with a short justification that cites specific chunk indices.\n"
        "- Evaluate each of the 4 auto-fail flags independently (triggered true/false, with details when true).\n"
        "- Do NOT compute weighted_score or overall_rating; those are calculated downstream.\n"
        "- Base every judgment only on the supplied source document and chunks."
    )


def evaluate_file_chunks(chunks: list[Document]) -> None:
    """Grade a single file's chunks against the rubric and write JSON + Markdown reports.

    The chunks carry the original file path in ``metadata['source']``; the source document is
    reloaded so the judge can assess completeness and boundary quality against the original.
    Results are written under ``backend/evals/results/`` mirroring the source directory tree.
    """
    if not chunks:
        print(f"{YELLOW}No chunks to evaluate; skipping.{RESET}")
        return

    source = chunks[0].metadata.get("source")
    if not source:
        print(f"{RED}Chunks are missing a 'source' in metadata; cannot evaluate.{RESET}")
        return

    source_document = load_document_at_path(str(source))
    if source_document is None:
        print(f"{RED}Could not reload source document at {source}; skipping evaluation.{RESET}")
        return

    rubric_text = RUBRIC_PATH.read_text(encoding="utf-8")
    prompt = _build_judge_prompt(rubric_text, source_document.page_content, chunks)

    system_prompt = (
        "You are a meticulous RAG chunking evaluator. You grade how well a document was split "
        "into retrieval chunks, scoring strictly and only against the provided rubric. Be concrete, "
        "cite chunk indices as evidence, and never invent content that is not in the supplied material."
    )

    llm = ChatAnthropic(model=JUDGE_MODEL, temperature=0.0).with_structured_output(ChunkingAssessment)
    assessment = llm.invoke([
        ("system", system_prompt),
        ("human", prompt),
    ])

    result = _build_assessment_dict(
        assessment=assessment,
        source=str(source),
        source_length_chars=len(source_document.page_content),
        chunk_count=len(chunks),
    )

    json_path, md_path = _assessment_output_paths(str(source))
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown_report(result), encoding="utf-8")

    rating = result["overall_rating"]
    if rating in ("Excellent", "Good"):
        rating_color = GREEN
    elif rating == "Acceptable":
        rating_color = YELLOW
    else:
        rating_color = RED

    print(
        f"Assessed {Path(source).name}: "
        f"{rating_color}{rating} ({result['weighted_score']}/5){RESET} -> {json_path.relative_to(EVALS_DIR.parent)}"
    )
