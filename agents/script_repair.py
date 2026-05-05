from typing import Dict, List

from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.screenplay_format import strip_markdown_fences


def script_repair_agent(state):
    inv = list(state.get("llm_invocations") or [])
    count = int(state.get("script_repair_count") or 0) + 1
    script = str(state.get("script", "") or "")
    report = state.get("validation_report") or {}

    issues: List[str] = list(report.get("issues") or [])
    suggestions: List[str] = list(report.get("suggestions") or [])
    llm_notes = report.get("llm_notes")
    if isinstance(llm_notes, dict):
        li = llm_notes.get("issues") or []
        ls = llm_notes.get("suggestions") or []
        if isinstance(li, list):
            issues.extend(str(x) for x in li)
        if isinstance(ls, list):
            suggestions.extend(str(x) for x in ls)

    print(
        f"\nScript did not pass structural validation — auto-repair attempt {count}/2 "
        "(rewriting from validator feedback)...\n"
    )

    if not llm_configured():
        return {"script_repair_count": count, "llm_invocations": inv}

    llm = get_chat_llm(temperature=0.25)
    if llm is None:
        return {"script_repair_count": count, "llm_invocations": inv}

    meta = describe_llm(llm)
    issue_block = "\n".join(f"- {i}" for i in issues) if issues else "- (none listed)"
    hint_block = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- (none listed)"

    prompt = f"""Rewrite the screenplay below so it passes a strict structural checker.

Required format (plain text only — no markdown, no **bold**, no ``` fences):
1) Scene headings: e.g. "Scene 1 - EXT. PARK - DAY" or lines starting with INT. or EXT.
2) Action lines in parentheses, e.g. (Rain falls on empty paths.)
3) Dialogue lines: CHARACTER IN ALL CAPS: spoken text
   Optional parenthetical before colon: JESSICA (V.O.): whispered line
4) At least two scenes and multiple dialogue lines when it fits the story.

Original creative brief (keep the same story and tone):
{state.get("input_prompt", "")}

Current script:
---
{script}
---

Reported problems:
{issue_block}

Suggested fixes:
{hint_block}

Return ONLY the full revised screenplay. No title line, no commentary."""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", "")
        if isinstance(content, str):
            fixed = strip_markdown_fences(content)
            if fixed:
                inv.append({"step": "script_repair", **meta})
                return {
                    "script": fixed,
                    "script_repair_count": count,
                    "llm_invocations": inv,
                }
    except Exception:
        pass

    return {"script_repair_count": count, "llm_invocations": inv}
