"""
Config load/save with Pydantic. Singleton pattern via module-level instance.
Config file lives at ~/.voice_os/config.json; falls back to defaults on first run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".voice_os"
_CONFIG_PATH = _CONFIG_DIR / "config.json"


class DefaultApps(BaseModel):
    browser: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None
    pdf: Optional[str] = None
    docx: Optional[str] = None
    xls: Optional[str] = None
    image: Optional[str] = None
    archive: Optional[str] = None


class Settings(BaseModel):
    mic_device_id: Optional[int] = None
    wake_phrase: str = Field(default="OS Assistant", max_length=25)
    llm_model: str = "mistral"
    tts_engine: str = Field(default="kokoro", pattern="^(kokoro|pyttsx3)$")
    whitelisted_dirs: List[str] = Field(default_factory=list)
    default_apps: DefaultApps = Field(default_factory=DefaultApps)
    ollama_base_url: str = "http://localhost:11434"
    ui_port: int = Field(default=7860, ge=1024, le=65535)

    @field_validator("wake_phrase")
    @classmethod
    def wake_phrase_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("wake_phrase must not be blank")
        return v.strip()

    @field_validator("whitelisted_dirs")
    @classmethod
    def resolve_dirs(cls, dirs: List[str]) -> List[str]:
        resolved = []
        for d in dirs:
            p = Path(d).expanduser().resolve()
            if p.is_dir():
                resolved.append(str(p))
            else:
                logger.warning("Whitelisted dir does not exist, skipping: %s", d)
        return resolved

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "Settings":
        """Load from ~/.voice_os/config.json, or return defaults."""
        if _CONFIG_PATH.exists():
            try:
                data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
                return cls.model_validate(data)
            except Exception as exc:
                logger.warning("Failed to parse config (%s); using defaults.", exc)
        return cls()

    def save(self) -> None:
        """Persist current settings to ~/.voice_os/config.json."""
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info("Config saved to %s", _CONFIG_PATH)

    def update(self, **kwargs) -> None:
        """Update fields in-place and save.  Caller may pass nested dicts."""
        data = self.model_dump()
        data.update(kwargs)
        validated = Settings.model_validate(data)
        # Replace own fields with validated values
        for field in self.model_fields:
            setattr(self, field, getattr(validated, field))
        self.save()


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere:
#   from voice_os.config.settings import settings
# ---------------------------------------------------------------------------
settings = Settings.load()
