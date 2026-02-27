"""Hints loader: reads config/hints.yaml and injects tool hints into user messages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Compiled patterns for extracting an explicit repo/directory name from a message.
# Matches common Portuguese and English preposition phrases like:
#   "do repo gh2", "no repo gh2", "of repo gh2", "in repo gh2"
_REPO_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r'\bdo\s+repo\s+([\w][\w.-]*)', re.IGNORECASE),
    re.compile(r'\bno\s+repo\s+([\w][\w.-]*)', re.IGNORECASE),
    re.compile(r'\bdo\s+reposit[oó]rio\s+([\w][\w.-]*)', re.IGNORECASE),
    re.compile(r'\bno\s+reposit[oó]rio\s+([\w][\w.-]*)', re.IGNORECASE),
    re.compile(r'\bof\s+repo\s+([\w][\w.-]*)', re.IGNORECASE),
    re.compile(r'\bin\s+repo\s+([\w][\w.-]*)', re.IGNORECASE),
    re.compile(r'\bfrom\s+repo\s+([\w][\w.-]*)', re.IGNORECASE),
]


def _extract_repo(message: str) -> Optional[str]:
    """Extract a repo/directory name if the user referenced one explicitly.

    Examples:
        "liste arquivos do repo gh2"  → "gh2"
        "git status no repo my-app"   → "my-app"
        "list files in repo frontend" → "frontend"
    """
    for pattern in _REPO_PATTERNS:
        m = pattern.search(message)
        if m:
            return m.group(1)
    return None


class HintRule:
    """A single hint rule loaded from hints.yaml."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.description: str = data.get("description", "")
        self.patterns: List[str] = [p.lower() for p in data.get("patterns", [])]
        # Prefer new name; fall back to old name for backward compat
        self.requires_no_abspath: bool = (
            data.get("requires_no_abspath", data.get("requires_no_path", False))
        )
        self.inject: str = data.get("inject", "")

    def matches(self, message: str) -> bool:
        msg = message.lower().strip()
        if self.requires_no_abspath:
            # Block only if the message contains an explicit absolute container path
            # (e.g. /projects/foo or /workspace/bar).  Tilde paths like ~/repos are fine.
            import re
            if re.search(r'(?<![~\w])/(?:projects|workspace)\b', message):
                return False
        return any(
            msg == p or msg.startswith(p + " ") or (" " + p) in msg or msg.endswith(" " + p)
            for p in self.patterns
        )

    def build_hint(
        self,
        filesystem_root: str = "/projects",
        workspace_root: str = "/workspace",
        message: str = "",
    ) -> str:
        """Build the hint text, optionally scoped to a repo extracted from the message.

        If the user mentioned a specific repo (e.g. "do repo gh2"), the hint will
        include the full path (/projects/gh2) instead of just the root (/projects).
        """
        repo = _extract_repo(message) if message else None
        if repo:
            filesystem_root = f"{filesystem_root}/{repo}"
            workspace_root = f"{workspace_root}/{repo}"
        return self.inject.format(
            filesystem_root=filesystem_root,
            workspace_root=workspace_root,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "patterns": self.patterns,
            "requires_no_abspath": self.requires_no_abspath,
            "inject": self.inject,
        }


class HintsLoader:
    """Loads hint rules from a YAML file and applies them to user messages."""

    def __init__(self, hints_path: str = "config/hints.yaml") -> None:
        self.hints_path = Path(hints_path)
        self.rules: List[HintRule] = self._load()

    def _load(self) -> List[HintRule]:
        if not self.hints_path.exists():
            return []
        with open(self.hints_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return [HintRule(r) for r in data.get("hints", [])]

    def reload(self) -> None:
        """Reload hints from disk (useful after file changes)."""
        self.rules = self._load()

    def match(
        self,
        message: str,
        filesystem_root: str = "/projects",
        workspace_root: str = "/workspace",
    ) -> Optional[str]:
        """Return the first matching hint text, or None if no rule matches."""
        for rule in self.rules:
            if rule.matches(message):
                return rule.build_hint(filesystem_root, workspace_root, message=message)
        return None

    def apply(
        self,
        message: str,
        filesystem_root: str = "/projects",
        workspace_root: str = "/workspace",
    ) -> str:
        """Return message with hint appended if a rule matches, otherwise unchanged."""
        hint = self.match(message, filesystem_root, workspace_root)
        if hint:
            return f"{message}\n[{hint}]"
        return message

    def all_hints(self) -> List[Dict[str, Any]]:
        """Return all loaded rules as dicts (for API exposure)."""
        return [r.to_dict() for r in self.rules]


# Module-level singleton (lazy-loaded)
_loader: Optional[HintsLoader] = None


def get_hints_loader(hints_path: str = "config/hints.yaml") -> HintsLoader:
    global _loader
    if _loader is None:
        _loader = HintsLoader(hints_path)
    return _loader
