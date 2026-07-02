"""Download-on-first-run access to the prebuilt ChromaDB index.

The published package ships no vectors: the ~34 MB index lives as a GitHub Release asset and is
fetched + cached the first time the server needs it. Subsequent runs read straight from the
cache and work offline. Only the standard library is used for the download so the wheel stays
dependency-light (no httpx/requests).
"""

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import chromadb

from .config import COLLECTION_NAME, INDEX_URL, datastore_dir


# Sentinel file that marks a fully-extracted index. We only treat the cache as populated when
# this exists, so an interrupted download/extraction never leaves a half-written index that
# looks valid on the next run.
_MARKER = "chroma.sqlite3"


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checksum(archive: Path, url: str) -> None:
    """Verify `archive` against its published `<url>.sha256`, if one is available.

    The checksum file contents may be a bare hex digest or the common `"<digest>  <filename>"`
    format; we take the first whitespace-delimited token either way. A missing checksum file is
    tolerated (some releases may not publish one) but a present-and-mismatched one is fatal.
    """
    try:
        with urllib.request.urlopen(url + ".sha256") as response:
            expected = response.read().decode().strip().split()[0]
    except Exception:
        return

    actual = _sha256(archive)
    if actual != expected:
        raise RuntimeError(
            f"Index checksum mismatch: expected {expected}, got {actual}. "
            f"Refusing to use a corrupted download from {url}."
        )


def ensure_index() -> Path:
    """Return the local index directory, downloading + extracting it on first use.

    Extraction is staged in a sibling temp directory and atomically renamed into place so a
    concurrent or interrupted run can't expose a partially-written index.
    """
    dest = datastore_dir()
    if (dest / _MARKER).exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "index.tar.gz"

        _download(INDEX_URL, archive)
        _verify_checksum(archive, INDEX_URL)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(archive) as tar:
            _safe_extractall(tar, extract_dir)

        # Support archives that either contain the store files at the top level or nest them
        # under a single wrapping directory.
        root = extract_dir
        if not (root / _MARKER).exists():
            subdirs = [p for p in root.iterdir() if p.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / _MARKER).exists():
                root = subdirs[0]
        if not (root / _MARKER).exists():
            raise RuntimeError(
                f"Downloaded index archive from {INDEX_URL} did not contain '{_MARKER}'."
            )

        # Another process may have won the race and populated dest while we were downloading.
        if (dest / _MARKER).exists():
            return dest
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(root, dest)

    return dest


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract `tar` into `dest`, rejecting members that would escape the target directory.

    Guards against path-traversal ("tar slip") in a downloaded archive by resolving each
    member's destination and confirming it stays within `dest`.
    """
    dest_root = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not (target == dest_root or dest_root in target.parents):
            raise RuntimeError(f"Unsafe path in index archive: {member.name!r}")
    tar.extractall(dest)


def get_rag_collection():
    """Return the persistent ChromaDB collection, ensuring the index is present first."""
    client = chromadb.PersistentClient(path=str(ensure_index()))
    return client.get_collection(name=COLLECTION_NAME)
