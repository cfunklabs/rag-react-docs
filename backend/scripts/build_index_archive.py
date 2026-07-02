"""Package the local ChromaDB index into a release asset for download-on-first-run.

Dev-side build step (not shipped in the wheel). After rebuilding the local index with
`uv run main.py`, run this to produce a versioned tarball plus a sha256 sidecar, ready to
upload as a GitHub Release asset. The published `cfunklabs-rag-react-docs` package downloads
and verifies exactly these files (see src/rag_react_docs/datastore.py).

Usage:
    uv run scripts/build_index_archive.py            # uses INDEX_VERSION from the package
    uv run scripts/build_index_archive.py --version 19-2-v2

Then upload both files to a release tagged `index-<version>`, e.g.:
    gh release create index-19-2-v1 dist/rag-index-19-2-v1.tar.gz dist/rag-index-19-2-v1.tar.gz.sha256
"""

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASTORE_DIR = BACKEND_DIR / "rag_datastore"
OUTPUT_DIR = BACKEND_DIR / "dist"
MARKER = "chroma.sqlite3"


def _default_version() -> str:
    # Import lazily so the script still runs without the package installed on the path.
    sys.path.insert(0, str(BACKEND_DIR / "src"))
    from rag_react_docs.config import INDEX_VERSION

    return INDEX_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(version: str) -> Path:
    if not (DATASTORE_DIR / MARKER).exists():
        raise SystemExit(
            f"No index found at {DATASTORE_DIR} (missing '{MARKER}'). "
            f"Run 'uv run main.py' to build the index first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT_DIR / f"rag-index-{version}.tar.gz"

    # Archive the *contents* of rag_datastore at the top level (chroma.sqlite3 + the hnswlib
    # index dir) so extraction yields a directory that PersistentClient can open directly.
    with tarfile.open(archive, "w:gz") as tar:
        for item in sorted(DATASTORE_DIR.iterdir()):
            if item.name == ".DS_Store":
                continue
            tar.add(item, arcname=item.name)

    checksum = _sha256(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive.name}\n")

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"Wrote {archive} ({size_mb:.1f} MB)")
    print(f"Wrote {checksum_path} ({checksum})")
    print()
    print("Next: upload both files to a release tagged 'index-" + version + "', e.g.")
    print(f"  gh release create index-{version} {archive} {checksum_path}")
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=None,
        help="Index version tag (defaults to INDEX_VERSION in rag_react_docs.config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = args.version or _default_version()
    build(version)


if __name__ == "__main__":
    main()
