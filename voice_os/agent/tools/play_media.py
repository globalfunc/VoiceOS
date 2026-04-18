"""
LangChain tool: play_media

Searches the user's whitelisted directories for media files matching the
given title, then launches them with the configured player or xdg-open.

Fuzzy matching handles STT mis-transcriptions and partial titles.
When the best match is ambiguous, the tool lists alternatives so the LLM
can relay them to the user.

Media extensions recognised
───────────────────────────
  Video: .mp4 .mkv .avi .mov .wmv .webm .m4v .flv .ts
  Audio: .mp3 .flac .wav .ogg .m4a .aac .wma .opus
"""
from __future__ import annotations

import difflib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Type

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_VIDEO_EXTS: frozenset = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v", ".flv", ".ts",
})
_AUDIO_EXTS: frozenset = frozenset({
    ".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".wma", ".opus",
})
_MEDIA_EXTS: frozenset = _VIDEO_EXTS | _AUDIO_EXTS

# Fuzzy match thresholds.
_AUTO_PLAY_SCORE  = 0.6   # play without asking if top match >= this
_MIN_SCORE        = 0.3   # discard candidates below this
_MAX_CANDIDATES   = 5


class _PlayMediaInput(BaseModel):
    title: str = Field(
        description=(
            "Title or name of the media to play, e.g. 'The Godfather', "
            "'bohemian rhapsody', 'podcast episode 42'. Partial matches are fine."
        )
    )
    application: Optional[str] = Field(
        default=None,
        description=(
            "Specific media player to use, e.g. 'vlc', 'mpv'. "
            "If omitted, the configured default or xdg-open is used."
        ),
    )


class PlayMediaTool(BaseTool):
    """Play a media file (video or audio) by title from whitelisted directories."""

    name: str = "play_media"
    description: str = (
        "Play a media file (video or audio) by title from the user's whitelisted directories. "
        "Fuzzy-matches partial titles and STT mis-transcriptions. "
        "Optionally specify which media player application to use."
    )
    args_schema: Type[BaseModel] = _PlayMediaInput

    def _run(self, title: str, application: Optional[str] = None) -> str:  # type: ignore[override]
        from voice_os.config.settings import settings

        dirs = settings.whitelisted_dirs
        if not dirs:
            return (
                "No directories are whitelisted. "
                "Please add your media directories in settings first."
            )

        title = title.strip()
        candidates = _find_media(title, dirs)

        if not candidates:
            return (
                f"Could not find any media matching '{title}' "
                "in your whitelisted directories."
            )

        best_path, best_score, best_name = candidates[0]

        # Strong match or only one candidate → play immediately.
        if best_score >= _AUTO_PLAY_SCORE or len(candidates) == 1:
            return self._play(best_path, best_name, application, settings)

        # Ambiguous → list the top options for the user to choose from.
        options = [c[2] for c in candidates[:3]]
        options_str = ", ".join(f"'{n}'" for n in options)
        return (
            f"I found a few possible matches: {options_str}. "
            "Please say 'play' followed by the exact title."
        )

    async def _arun(self, title: str, application: Optional[str] = None) -> str:  # type: ignore[override]
        return self._run(title, application)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _play(
        self,
        path: str,
        display_name: str,
        application: Optional[str],
        settings,
    ) -> str:
        ext = Path(path).suffix.lower()
        player = _resolve_player(ext, application, settings)

        try:
            cmd = [player, path]
            subprocess.Popen(
                cmd,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            player_label = application or player
            logger.info("play_media: launched '%s' with '%s'.", path, player_label)
            return f"Playing {display_name} using {player_label}."
        except FileNotFoundError:
            logger.warning("play_media: player '%s' not found, falling back to xdg-open.", player)
            if player != "xdg-open":
                try:
                    subprocess.Popen(
                        ["xdg-open", path],
                        start_new_session=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return (
                        f"Playing {display_name}. "
                        f"Note: {player} was not found, used the system default."
                    )
                except Exception as exc2:
                    logger.error("play_media xdg-open fallback: %s", exc2)
            return f"Could not open {display_name}: player '{player}' not found."
        except Exception as exc:
            logger.error("play_media._play: %s", exc)
            return f"Failed to play {display_name}: {exc}"


# ---------------------------------------------------------------------------
# File discovery and fuzzy ranking
# ---------------------------------------------------------------------------

def _find_media(title: str, dirs: List[str]) -> List[Tuple[str, float, str]]:
    """
    Walk whitelisted directories for media files and fuzzy-rank them against
    ``title``.

    Returns ``[(path, score, display_name), ...]`` sorted by score descending.
    Only files with score >= ``_MIN_SCORE`` are included.
    """
    title_norm = _norm(title)
    scored: List[Tuple[str, float, str]] = []

    for directory in dirs:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            continue
        try:
            for root, _subdirs, files in os.walk(directory):
                for filename in files:
                    ext = Path(filename).suffix.lower()
                    if ext not in _MEDIA_EXTS:
                        continue
                    stem = Path(filename).stem
                    stem_norm = _norm(stem)

                    score = difflib.SequenceMatcher(
                        None, title_norm, stem_norm
                    ).ratio()

                    # Substring bonus: query appears inside the stem.
                    if title_norm and title_norm in stem_norm:
                        score = max(score, 0.7)

                    if score >= _MIN_SCORE:
                        full = str(Path(root) / filename)
                        scored.append((full, score, stem))
        except PermissionError:
            logger.debug("_find_media: permission denied on %s", directory)

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:_MAX_CANDIDATES]


def _norm(s: str) -> str:
    """Collapse whitespace, dashes, underscores, and dots; lower-case."""
    return re.sub(r"[\s\-_.]", "", s).lower()


def _resolve_player(
    ext: str, application: Optional[str], settings
) -> str:
    """Return the media player to use, in preference order."""
    if application:
        return application
    da = settings.default_apps
    if ext in _VIDEO_EXTS and da.video:
        return da.video
    if ext in _AUDIO_EXTS and da.audio:
        return da.audio
    return "xdg-open"
