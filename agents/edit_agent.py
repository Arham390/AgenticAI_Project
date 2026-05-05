"""Phase 5 — Intelligent Edit & Undo Agent.

Accepts a free-text edit instruction, classifies intent via LLM (with a
rule-based fallback), and executes the targeted phase re-run.  Every edit is
snapshotted so the user can undo any number of times.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.llm_factory import describe_llm, get_chat_llm, llm_configured

# ─────────────────────────────────────────────────────────────────────────────
# Intent schema
# ─────────────────────────────────────────────────────────────────────────────

VALID_INTENTS = {
    "change_voice_tone": "audio",
    "add_background_music": "audio",
    "change_voice_speed": "audio",
    "re_synthesize_audio": "audio",
    "make_scene_darker": "video_frame",
    "make_scene_brighter": "video_frame",
    "change_character_design": "video_frame",
    "change_visual_style": "video_frame",
    "re_generate_visuals": "video_frame",
    "remove_subtitle": "video",
    "speed_up_scene": "video",
    "slow_down_scene": "video",
    "recompose_video": "video",
    "regenerate_script": "script",
    "change_scene_dialogue": "script",
    "add_scene": "script",
    "unknown": "unknown",
}

INTENT_KEYWORDS: List[tuple[str, str]] = [
    (r"voice|tone|speak|narrat|tts|audio|sound", "re_synthesize_audio"),
    (r"music|background|bgm|soundtrack|score", "add_background_music"),
    (r"dark|bright|light|color|colour|visual|style|look|aesthetic", "change_visual_style"),
    (r"character.design|appearance|redesign|character look", "change_character_design"),
    (r"subtitle|caption|text.overlay", "remove_subtitle"),
    (r"speed up|faster|slow|slower|timing", "speed_up_scene"),
    (r"script|story|dialogue|rewrite|regenerate", "regenerate_script"),
    (r"scene|frame|image|picture|visual|render", "re_generate_visuals"),
    (r"video|compose|export|mp4|output", "recompose_video"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_intent(query: str) -> Dict[str, Any]:
    """Fallback: keyword pattern matching."""
    q = query.lower()
    for pattern, intent in INTENT_KEYWORDS:
        if re.search(pattern, q):
            target = VALID_INTENTS.get(intent, "unknown")
            return {
                "intent": intent,
                "target": target,
                "scope": "all",
                "parameters": {"raw_query": query},
                "confidence": "rule_based",
            }
    return {
        "intent": "unknown",
        "target": "unknown",
        "scope": "all",
        "parameters": {"raw_query": query},
        "confidence": "rule_based",
    }


def _llm_intent(query: str) -> Optional[Dict[str, Any]]:
    """LLM-powered intent classification."""
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=0)
    if llm is None:
        return None

    valid_intents_str = ", ".join(VALID_INTENTS.keys())
    valid_targets = "audio | video_frame | video | script | unknown"
    system_prompt = f"""You are an intent classification agent for a video editing pipeline.

Classify the user's edit query into a structured JSON intent object.
Valid intents: {valid_intents_str}
Valid targets: {valid_targets}

Output ONLY valid JSON matching this schema exactly:
{{
  "intent": "<one of the valid intents>",
  "target": "<audio|video_frame|video|script|unknown>",
  "scope": "<all|character:NAME|scene:ID>",
  "parameters": {{<any relevant key-value pairs>}}
}}

User query: {query}"""

    try:
        response = llm.invoke(system_prompt)
        content = getattr(response, "content", "") or ""
        # Extract JSON from response.
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "intent" in parsed:
                parsed["confidence"] = "llm"
                return parsed
    except Exception:
        pass
    return None


def classify_intent(query: str) -> Dict[str, Any]:
    """Classify edit query. Tries LLM first, falls back to rule-based."""
    result = _llm_intent(query)
    if result is None:
        result = _rule_based_intent(query)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Edit executors
# ─────────────────────────────────────────────────────────────────────────────

def _execute_audio_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-synthesise voice audio for scenes affected by the edit."""
    from agents.voice_synth import voice_synth_agent

    base = {
        "scene_manifest_data": state.get("scene_manifest_data", {}),
        "audio_tracks": [],
        "llm_invocations": [],
    }
    result = voice_synth_agent(base)
    return {"audio_edit_applied": True, "audio_tracks": result.get("audio_tracks", []), **result}


def _execute_video_frame_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-generate video frames for affected scenes."""
    from agents.video_gen import video_gen_agent

    params = intent.get("parameters", {})
    scene_manifest = state.get("scene_manifest_data", {})
    if isinstance(scene_manifest, dict):
        scenes = scene_manifest.get("scenes", [])
        for scene in scenes:
            if "style" not in scene:
                scene["style"] = "cinematic"
            raw_q = (params.get("raw_query") or "").lower()
            if "dark" in raw_q:
                scene["style"] = "dark moody cinematic low-key lighting"
            elif "bright" in raw_q:
                scene["style"] = "bright vivid high-key lighting"
            elif "style" in raw_q or "aesthetic" in raw_q:
                scene["style"] = params.get("style", "artistic cinematic")

    base = {
        "scene_manifest_data": scene_manifest,
        "video_tracks": [],
        "images": state.get("images", []),
        "llm_invocations": [],
    }
    result = video_gen_agent(base)
    return {"video_frame_edit_applied": True, **result}


def _execute_video_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Recompose final video with updated parameters."""
    from agents.lip_sync import lip_sync_agent

    base = {
        "scene_manifest_data": state.get("scene_manifest_data", {}),
        "audio_tracks": state.get("audio_tracks", []),
        "video_tracks": state.get("video_tracks", []),
        "face_swaps": state.get("face_swaps", []),
        "scene_tasks": state.get("scene_tasks", []),
        "raw_scenes": [],
    }
    result = lip_sync_agent(base)
    return {"video_edit_applied": True, **result}


def _execute_script_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run the scriptwriter and cascade through character + image agents."""
    params = intent.get("parameters", {})
    prompt = (params.get("prompt") or params.get("raw_query") or
              "Regenerate the script with improvements.")

    from agents.scriptwriter import scriptwriter_agent
    from agents.validator import validator_agent
    from agents.character import character_agent
    from agents.image import image_agent
    from agents.scene_parser import scene_parser_agent

    s = {
        "input_prompt": prompt, "manual_script": "", "mode": "autonomous",
        "validated": False, "validation_report": {}, "characters": [],
        "images": [], "scene_manifest_data": {}, "scene_tasks": [],
        "audio_tracks": [], "video_tracks": [], "face_swaps": [],
        "raw_scenes": [], "llm_invocations": [], "script_repair_count": 0,
        "require_hitl": False, "approved": False, "script": "",
    }
    s.update(scriptwriter_agent(s))
    s.update(validator_agent(s))
    s.update(character_agent(s))
    s.update(image_agent(s))
    s.update(scene_parser_agent(s))

    # Persist script.
    script_path = _PROJECT_ROOT / "outputs" / "script.txt"
    script_path.write_text(s.get("script", ""), encoding="utf-8")

    return {"script_edit_applied": True, **s}


_EXECUTORS = {
    "audio": _execute_audio_edit,
    "video_frame": _execute_video_frame_edit,
    "video": _execute_video_edit,
    "script": _execute_script_edit,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def process_edit(
    query: str,
    current_state: Dict[str, Any],
    state_manager: Any,
) -> Dict[str, Any]:
    """Classify the edit query, execute it, and save a versioned snapshot.

    Returns a summary dict with keys: intent, edit_result, version.
    """
    intent = classify_intent(query)
    target = intent.get("target", "unknown")

    executor = _EXECUTORS.get(target)
    if executor is None:
        return {
            "intent": intent,
            "edit_result": {"status": "skipped", "reason": f"Unknown target: {target}"},
            "version": None,
        }

    try:
        edit_result = executor(intent, current_state)
        merged = {**current_state, **edit_result}
        description = f"Edit [{intent.get('intent', '?')}]: {query[:80]}"
        version = state_manager.snapshot(merged, description=description)
        return {"intent": intent, "edit_result": edit_result, "version": version}
    except Exception as exc:
        return {
            "intent": intent,
            "edit_result": {"status": "error", "error": str(exc)},
            "version": None,
        }
