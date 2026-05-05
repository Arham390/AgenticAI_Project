import json
from typing import Dict, List, Optional, Tuple

from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.mcp_registry import invoke_tool
from tools.screenplay_format import rule_validate_structure


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return "{}"


def _try_llm_validation(script: str) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, str]]]:
    if not llm_configured():
        return None, None

    llm = get_chat_llm(temperature=0)
    if llm is None:
        return None, None

    meta = describe_llm(llm)
    prompt = f"""
    You are a Script Validator Agent (advisory only — another layer checks structure mechanically).

    Review the script for:
    - Scene headings (Scene N, INT., or EXT.)
    - Dialogue with CHARACTER: or CHARACTER (V.O.): style
    - Clear action lines in parentheses

    Be lenient: minor stylistic choices are fine if the script is readable and filmable.
    Set "valid" to true unless there are clear structural problems (no scenes, no dialogue, etc.).

    Return ONLY valid JSON:
    {{
        "valid": true/false,
        "issues": ["list of issues"],
        "suggestions": ["list of fixes"]
    }}

    Script:
    {script}
    """

    try:
        response = llm.invoke(prompt)
    except Exception:
        return None, None

    content = getattr(response, "content", "")
    if not isinstance(content, str):
        return None, None

    try:
        result = json.loads(_extract_json_object(content))
    except Exception:
        return None, None

    if not isinstance(result, dict):
        return None, None

    return (
        {
            "valid": bool(result.get("valid", False)),
            "issues": result.get("issues", []) if isinstance(result.get("issues", []), list) else [],
            "suggestions": result.get("suggestions", [])
            if isinstance(result.get("suggestions", []), list)
            else [],
        },
        meta,
    )


def validator_agent(state):
    script = state.get("script", "")
    inv = list(state.get("llm_invocations") or [])
    rule_result = rule_validate_structure(script)
    llm_result, llm_meta = _try_llm_validation(script)
    if llm_meta:
        inv.append({"step": "validator", **llm_meta})

    if llm_result is None:
        report = {
            "source": "rule-only",
            "valid": bool(rule_result.get("valid", False)),
            "issues": list(rule_result.get("issues", [])),
            "suggestions": list(rule_result.get("suggestions", [])),
        }
    else:
        report = {
            "source": "rule+llm_advisory",
            "valid": bool(rule_result.get("valid", False)),
            "issues": list(rule_result.get("issues", [])),
            "suggestions": list(rule_result.get("suggestions", [])),
            "llm_notes": {
                "agrees": bool(llm_result.get("valid", False)),
                "issues": llm_result.get("issues", []),
                "suggestions": llm_result.get("suggestions", []),
            },
        }

    result = {
        "validated": bool(report["valid"]),
        "validation_report": report,
    }

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "validator",
                    "validated": result["validated"],
                    "report": report,
                }
            },
        )
    except Exception:
        pass

    result["llm_invocations"] = inv
    return result