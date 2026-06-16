#!/usr/bin/env python3
"""Download every linked .md doc from the React llms.txt index.

Parses https://react.dev/llms.txt, follows each `https://react.dev/*.md`
link, and saves the destination markdown into the docs folder. The index
heading structure (excluding the top-level heading) becomes a directory tree,
and each file is named after its frontmatter title, e.g.:

    docs/API Reference/React/Components/Built-in React Components.md

Self-contained: standard library only.
"""

import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _build_ssl_context() -> ssl.SSLContext:
    """Build a verifying SSL context, using certifi's CA bundle if available.

    Some interpreters (notably macOS framework Python) ship without a usable
    system CA store, which breaks HTTPS verification. certifi provides a bundle.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _build_ssl_context()

INDEX_URL = "https://react.dev/llms.txt"
# scripts/ lives in rag_py/, but docs/ stays in the monorepo root (two levels up).
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs"
USER_AGENT = "react-docs-fetcher/1.0 (+https://react.dev)"

HEADING_RE = re.compile(r"^(#+)\s+(.+?)\s*$")
LINK_RE = re.compile(r"\[[^\]]*\]\((https://react\.dev/[^)]+\.md)\)")
TITLE_RE = re.compile(r'^title:\s*["\']?(.+?)["\']?\s*$')


def fetch(url: str) -> str:
    """Fetch a URL and return its decoded text body."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset)


def parse_index(text: str) -> list[tuple[list[str], str]]:
    """Return (section_path, url) pairs for every .md link in the index.

    section_path is the list of active headings at level >= 2, in order,
    so the top-level (level 1) heading is excluded.
    """
    stack: dict[int, str] = {}
    results: list[tuple[list[str], str]] = []

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            stack[level] = heading.group(2)
            # Drop any deeper headings now out of scope.
            for deeper in [lvl for lvl in stack if lvl > level]:
                del stack[deeper]
            continue

        for url in LINK_RE.findall(line):
            section_path = [stack[lvl] for lvl in sorted(stack) if lvl >= 2]
            results.append((section_path, url))

    return results


def extract_title(content: str) -> str | None:
    """Extract the title from a document's leading `---` frontmatter block."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = TITLE_RE.match(line)
        if match:
            return match.group(1)
    return None


def slug_from_url(url: str) -> str:
    """Fallback name derived from the URL's filename (without .md)."""
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md")


def sanitize(part: str) -> str:
    """Make a path component safe for the filesystem."""
    part = part.replace("/", "-").replace(":", "-")
    part = re.sub(r"\s+", " ", part).strip()
    return part


def build_relative_path(section_path: list[str], title: str) -> Path:
    """Map section headings to directories and the title to the filename."""
    parts = [sanitize(p) for p in section_path]
    return Path(*parts, sanitize(title) + ".md")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching index: {INDEX_URL}")
    try:
        index_text = fetch(INDEX_URL)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"Failed to fetch index: {exc}", file=sys.stderr)
        return 1

    links = parse_index(index_text)
    total = len(links)
    print(f"Found {total} linked documents. Saving to: {OUTPUT_DIR}\n")

    succeeded = 0
    failures: list[tuple[str, str]] = []

    for i, (section_path, url) in enumerate(links, start=1):
        print(f"[{i}/{total}] {url}")
        try:
            content = fetch(url)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"    ! download failed: {exc}")
            failures.append((url, str(exc)))
            continue

        title = extract_title(content) or slug_from_url(url)
        relative_path = build_relative_path(section_path, title)
        destination = OUTPUT_DIR / relative_path
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        except OSError as exc:
            print(f"    ! write failed: {exc}")
            failures.append((url, str(exc)))
            continue

        print(f"    -> {relative_path}")
        succeeded += 1

    print(f"\nDone. {succeeded}/{total} saved to {OUTPUT_DIR}.")
    if failures:
        print(f"{len(failures)} failed:")
        for url, reason in failures:
            print(f"  - {url}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
