"""Build human-readable provenance labels from a chunk's stored metadata.

Ported from the dev tooling's `format_source_label`. The installed package has no repo `docs/`
directory, so the relative-path shortening will not match; in that case we fall back to the
source file's basename, still prefixed onto the stored heading hierarchy.
"""

from pathlib import Path


def format_source_label(metadata: dict) -> str:
    """Return a label like "memo.md > Reference > memo(Component, arePropsEqual?)".

    Combines the source file (shortened to its basename) with the stored heading hierarchy.
    """
    source = metadata.get("source", "unknown source")
    source = Path(source).name

    headings = [
        metadata[field]
        for field in ("Header 1", "Header 2", "Header 3")
        if metadata.get(field)
    ]
    if headings:
        return f"{source} > {' > '.join(headings)}"
    return source
