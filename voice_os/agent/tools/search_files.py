"""
LangChain tool: search_files

Searches the user's whitelisted directories for files by name or content.

search_type modes
─────────────────
  "name"    — walk whitelisted dirs and match filenames (wildcards supported)
  "content" — semantic search via ChromaDB (requires prior indexing)
  "auto"    — try name first; if fewer than 3 results, also try content search
              and merge results without duplicates
"""
from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import List, Type

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5


class _SearchFilesInput(BaseModel):
    query: str = Field(
        description=(
            "The filename pattern or topic to search for. "
            "For name search, wildcards are supported (e.g. '*.pdf', 'budget*'). "
            "For content search, use a natural language phrase such as "
            "'notes about the Q3 budget'."
        )
    )
    search_type: str = Field(
        default="auto",
        description=(
            "Search mode: "
            "'name' — fast filename match (wildcards OK), "
            "'content' — semantic search over indexed file content, "
            "'auto' — try name first, fall back to content if fewer than 3 results."
        ),
    )


class SearchFilesTool(BaseTool):
    """Find files by name pattern or content in the user's whitelisted directories."""

    name: str = "search_files"
    description: str = (
        "Find files by name or content in the user's whitelisted directories. "
        "Use search_type='name' for filename patterns (supports wildcards like '*.pdf'). "
        "Use search_type='content' for semantic content search (requires prior indexing). "
        "Use search_type='auto' (default) to try both automatically."
    )
    args_schema: Type[BaseModel] = _SearchFilesInput

    def _run(self, query: str, search_type: str = "auto") -> str:  # type: ignore[override]
        from voice_os.config.settings import settings

        dirs = settings.whitelisted_dirs
        if not dirs:
            return (
                "No directories are whitelisted. "
                "Please add directories in the settings panel first."
            )

        query = query.strip()
        search_type = search_type.lower().strip()
        results: List[str] = []

        if search_type in ("name", "auto"):
            results = _name_search(query, dirs)

        if search_type == "content" or (search_type == "auto" and len(results) < 3):
            content_hits = _content_search(query)
            seen = set(results)
            for path in content_hits:
                if path not in seen:
                    results.append(path)
                    seen.add(path)

        if not results:
            return f"No files found matching '{query}'."

        results = results[:_MAX_RESULTS]
        count = len(results)
        if count == 1:
            return f"I found one file: {results[0]}"
        joined = ", ".join(results[:-1]) + f", and {results[-1]}"
        return f"I found {count} files: {joined}"

    async def _arun(self, query: str, search_type: str = "auto") -> str:  # type: ignore[override]
        return self._run(query, search_type)


# ---------------------------------------------------------------------------
# Name-based search
# ---------------------------------------------------------------------------

def _name_search(query: str, dirs: List[str]) -> List[str]:
    """
    Walk ``dirs`` and return paths whose filenames match ``query``.

    Matching order (first match wins per file):
      1. Case-insensitive exact filename match.
      2. fnmatch glob — user may include * or ? wildcards; bare queries are
         wrapped as ``*query*`` for substring matching.
    """
    query_lower = query.lower()
    glob_pattern = query if ("*" in query or "?" in query) else f"*{query}*"

    matches: List[str] = []
    seen: set = set()

    for directory in dirs:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            continue
        try:
            for root, _subdirs, files in os.walk(directory):
                for filename in files:
                    fn_lower = filename.lower()
                    if (
                        fn_lower == query_lower
                        or fnmatch.fnmatch(fn_lower, glob_pattern.lower())
                    ):
                        full = str(Path(root) / filename)
                        if full not in seen:
                            seen.add(full)
                            matches.append(full)
                        if len(matches) >= _MAX_RESULTS:
                            return matches
        except PermissionError:
            logger.debug("_name_search: permission denied on %s", directory)

    return matches


# ---------------------------------------------------------------------------
# Content-based search (ChromaDB)
# ---------------------------------------------------------------------------

def _content_search(query: str) -> List[str]:
    """
    Semantic similarity search via the ChromaDB vector store.

    Returns a list of file paths ordered by relevance.
    Falls back to an empty list when the store is unavailable or empty.
    """
    try:
        from voice_os.memory.vector_store import get_vector_store

        store = get_vector_store()
        if not store.is_ready() or store.count() == 0:
            return []
        results = store.search(query, n_results=_MAX_RESULTS)
        return [r["path"] for r in results if "path" in r]
    except Exception as exc:
        logger.warning("_content_search: %s", exc)
        return []
