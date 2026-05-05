"""Shared structural checks and regexes for screenplay text."""

import re
from typing import Dict, List

_SCENE_HEADING = re.compile(r"(Scene\s+\d+|INT\.|EXT\.)", re.IGNORECASE)
_SCENE_HEADING_LINE = re.compile(
    r"^[ \t]*(?:Scene\s+\d+|INT\.|EXT\.)",
    re.IGNORECASE | re.MULTILINE,
)
# NAME: or NAME (V.O.): / NAME (CONT'D): etc.
_DIALOGUE = re.compile(
    r"^[ \t]*[A-Z][A-Z0-9_ ]*(\([^)]*\))?[ \t]*:[ \t]*\S",
    re.MULTILINE,
)
_ACTION = re.compile(r"\([^)]+\)")


def rule_validate_structure(script: str) -> Dict[str, object]:
    issues: List[str] = []
    suggestions: List[str] = []

    scene_count = len(list(_SCENE_HEADING_LINE.finditer(script)))

    if scene_count == 0:
        issues.append("Missing scene headings (Scene X, INT., or EXT.).")
        suggestions.append("Add clear scene headers before scene content.")
    elif scene_count < 2:
        issues.append("At least two scenes are required for a multi-scene screenplay.")
        suggestions.append("Add at least one additional scene heading and content.")
    if not _DIALOGUE.search(script):
        issues.append("Missing dialogue labels in CHARACTER: format.")
        suggestions.append(
            "Label dialogue like LEAD: line or JESSICA (V.O.): line — all caps name before the colon."
        )
    if not _ACTION.search(script):
        issues.append("Missing action or visual cue lines in parentheses.")
        suggestions.append("Add at least one action line like '(A door slams.)'.")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
    }


def strip_markdown_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
