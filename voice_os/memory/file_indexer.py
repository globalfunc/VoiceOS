"""
File indexer for VoiceOS — Phase 3.

Walks whitelisted directories, extracts a short text snippet from each file,
and upserts the results into the ChromaDB vector store so they can be found
via semantic search later.

Supported content extraction
─────────────────────────────
  Plain text   — .txt .md .py .json .csv .rst .yaml .yml .toml .ini .cfg
  PDF          — requires pypdf (or pdfminer.six as fallback)
  Word doc     — requires python-docx

All other file types (images, video, audio, binaries …) are indexed by
filename only; their text body is left empty so ChromaDB still knows they
exist and can surface them via name-based content search.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension sets
# ---------------------------------------------------------------------------

_TEXT_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".csv",
    ".rst", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh",
}
_PDF_EXTS  = {".pdf"}
_DOCX_EXTS = {".docx"}

# Media / binary — indexed by name only, no content extraction attempted.
_MEDIA_EXTS = {
    ".mp3", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v", ".flv",
    ".ts", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".wma", ".opus",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".heic",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".so", ".dll", ".bin", ".pyc", ".o", ".a",
}

_SNIPPET_CHARS = 500    # characters extracted per file for the embedding
_BATCH_SIZE    = 50     # documents sent to ChromaDB per upsert call


class FileIndexer:
    """
    Scans whitelisted directories and populates the VectorStore.

    Example usage::

        indexer = FileIndexer()
        n = indexer.index_dirs(
            ["/home/user/docs", "/home/user/media"],
            progress_callback=lambda msg: print(msg),
        )
        print(f"Indexed {n} files.")
    """

    def index_dirs(
        self,
        dirs: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        Index all files under ``dirs`` into the vector store.

        Args:
            dirs: Absolute paths to scan (should already be whitelisted).
            progress_callback: Optional callable called with status strings.

        Returns:
            Total number of files upserted.
        """
        from voice_os.memory.vector_store import get_vector_store

        store = get_vector_store()
        if not store.is_ready():
            msg = (
                "Vector store is unavailable — "
                "install chromadb and sentence-transformers to enable content search."
            )
            logger.warning(msg)
            if progress_callback:
                progress_callback(msg)
            return 0

        total = 0
        batch: List[dict] = []

        def _flush() -> None:
            nonlocal total
            if batch:
                store.upsert(batch)
                total += len(batch)
                batch.clear()

        for directory in dirs:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                logger.warning("FileIndexer: skipping missing directory: %s", directory)
                continue

            if progress_callback:
                progress_callback(f"Scanning {directory}…")

            for root, _subdirs, files in os.walk(directory):
                root_path = Path(root)
                # Skip hidden directories (e.g. .git, .cache).
                rel_parts = root_path.relative_to(dir_path).parts
                if any(p.startswith(".") for p in rel_parts):
                    _subdirs[:] = []   # prune walk
                    continue

                for filename in files:
                    # Skip hidden files.
                    if filename.startswith("."):
                        continue

                    file_path = root_path / filename
                    ext = file_path.suffix.lower()
                    text = self._extract_text(file_path, ext)

                    metadata: dict = {
                        "path":     str(file_path),
                        "filename": filename,
                        "ext":      ext,
                        "dir":      str(root_path),
                        "is_media": ext in _MEDIA_EXTS,
                    }
                    batch.append({
                        "path":     str(file_path),
                        # Always embed at least the filename so name-based
                        # semantic search works even for binary files.
                        "text":     text if text else filename,
                        "metadata": metadata,
                    })

                    if len(batch) >= _BATCH_SIZE:
                        _flush()
                        if progress_callback:
                            progress_callback(f"Indexed {total} files…")

        _flush()

        msg = f"Indexing complete — {total} files indexed."
        logger.info("FileIndexer: %s", msg)
        if progress_callback:
            progress_callback(msg)
        return total

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------

    def _extract_text(self, path: Path, ext: str) -> str:
        if ext in _TEXT_EXTS:
            return _read_plain_text(path)
        if ext in _PDF_EXTS:
            return _read_pdf(path)
        if ext in _DOCX_EXTS:
            return _read_docx(path)
        # Media, binary, or unknown — no text extraction.
        return ""


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_plain_text(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return fh.read(_SNIPPET_CHARS)
    except Exception as exc:
        logger.debug("_read_plain_text(%s): %s", path, exc)
        return ""


def _read_pdf(path: Path) -> str:
    # Try pypdf first (lighter dependency).
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            if len(text) >= _SNIPPET_CHARS:
                break
        return text[:_SNIPPET_CHARS]
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("_read_pdf pypdf(%s): %s", path, exc)

    # Fall back to pdfminer.six.
    try:
        from pdfminer.high_level import extract_text as pm_extract  # type: ignore
        text = pm_extract(str(path), maxpages=2) or ""
        return text[:_SNIPPET_CHARS]
    except ImportError:
        logger.debug(
            "pypdf and pdfminer.six not installed — PDF content not extracted. "
            "Install with: pip install pypdf"
        )
    except Exception as exc:
        logger.debug("_read_pdf pdfminer(%s): %s", path, exc)
    return ""


def _read_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
        doc = Document(str(path))
        text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
        return text[:_SNIPPET_CHARS]
    except ImportError:
        logger.debug(
            "python-docx not installed — .docx content not extracted. "
            "Install with: pip install python-docx"
        )
    except Exception as exc:
        logger.debug("_read_docx(%s): %s", path, exc)
    return ""
