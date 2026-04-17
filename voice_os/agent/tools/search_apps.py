"""
LangChain tool: search_apps

Searches the installed application catalog by name, description, or category.
Used when the user's request is ambiguous or descriptive:
  - "open a database app"
  - "open the SQL tool starting with H"
  - "show me web browsers"
  - "open something for editing images"

The LLM calls this tool, receives a natural-language list of matches, then
calls open_app with the chosen app name.
"""
from __future__ import annotations

import logging
from typing import Type

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _SearchAppsInput(BaseModel):
    query: str = Field(
        description=(
            "Search term — can be an app name, category, description, or partial name. "
            "Examples: 'database', 'browser', 'SQL client', 'starts with H', 'image editor'."
        )
    )
    top_n: int = Field(
        default=5,
        description="Maximum number of results to return (default 5).",
    )


class SearchAppsTool(BaseTool):
    """Search installed applications by name, description, or category."""

    name: str = "search_apps"
    description: str = (
        "Search installed applications by name, description, or category. "
        "Use this when the user describes an app vaguely or you are unsure which app they mean. "
        "Returns matching app names. Then call open_app with the chosen name."
    )
    args_schema: Type[BaseModel] = _SearchAppsInput

    def _run(self, query: str, top_n: int = 5) -> str:  # type: ignore[override]
        from voice_os.os_handlers import get_os_handler

        handler = get_os_handler()
        query = (query or "").strip()
        if not query:
            return "Please provide a search term."

        results = handler.search_apps(query, top_n=top_n)

        if not results:
            return f"No installed apps found matching '{query}'."

        if len(results) == 1:
            name = results[0][0]
            return f"Found one match: {name}."

        names = [r[0] for r in results]
        names_str = ", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 1 else names[0]
        return f"Found {len(names)} apps matching '{query}': {names_str}."

    async def _arun(self, query: str, top_n: int = 5) -> str:  # type: ignore[override]
        return self._run(query, top_n)
