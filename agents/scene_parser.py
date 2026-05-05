import json
import re
from pathlib import Path
from typing import Any, Dict, List

from tools.mcp_registry import invoke_tool


_SCENE_HEADING_RE = re.compile(r"^\s*(Scene\s+\d+|INT\.|EXT\.)", re.IGNORECASE)
_DIALOGUE_RE = re.compile(r"^\s*([A-Z][A-Z0-9_ ]{1,30})(\([^)]+\))?\s*:\s*(.+)$")


def _parse_script_to_scenes(script: str) -> List[Dict[str, Any]]:
    scenes: List[Dict[str, Any]] = []
    current_scene: Dict[str, Any] | None = None

    def ensure_scene() -> Dict[str, Any]:
        nonlocal current_scene
        if current_scene is None:
            current_scene = {
                "scene_id": f"scene_{len(scenes) + 1:02d}",
                "heading": f"Scene {len(scenes) + 1}",
                "dialogues": [],
                "actions": [],
                "raw_lines": [],
            }
            scenes.append(current_scene)
        return current_scene

    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _SCENE_HEADING_RE.search(line):
            current_scene = {
                "scene_id": f"scene_{len(scenes) + 1:02d}",
                "heading": line,
                "dialogues": [],
                "actions": [],
                "raw_lines": [],
            }
            scenes.append(current_scene)
            continue

        scene = ensure_scene()
        scene["raw_lines"].append(line)

        dialogue_match = _DIALOGUE_RE.match(line)
        if dialogue_match:
            scene["dialogues"].append(
                {
                    "character": dialogue_match.group(1).strip().title(),
                    "line": (dialogue_match.group(3) or "").strip(),
                }
            )
        elif line.startswith("(") and line.endswith(")"):
            scene["actions"].append(line)

    return scenes


def _load_manifest_from_disk() -> Dict[str, Any] | None:
    path = Path(__file__).resolve().parents[1] / "outputs" / "scene_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("scenes"), list):
        return payload
    return None


def _fallback_tasks_from_manifest(scene_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenes = scene_manifest.get("scenes", [])
    if not isinstance(scenes, list):
        return []

    tasks: List[Dict[str, Any]] = []
    for idx, scene in enumerate(scenes, start=1):
        scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
        dialogues = scene.get("dialogues", [])
        dialogue_count = len(dialogues) if isinstance(dialogues, list) else 0
        tasks.append(
            {
                "scene_id": scene_id,
                "task_audio": f"{scene_id}:voice",
                "task_video": f"{scene_id}:video",
                "task_face_swap": f"{scene_id}:face_swap",
                "task_lip_sync": f"{scene_id}:lip_sync",
                "dialogue_count": dialogue_count,
            }
        )
    return tasks


def scene_parser_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    scene_manifest = state.get("scene_manifest_data")
    has_embedded_scenes = isinstance(scene_manifest, dict) and isinstance(scene_manifest.get("scenes"), list)
    if not has_embedded_scenes:
        script = str(state.get("script", ""))
        if script.strip():
            scenes = _parse_script_to_scenes(script)
            scene_manifest = {
                "mode": state.get("mode", "autonomous"),
                "validated": bool(state.get("validated", False)),
                "scene_count": len(scenes),
                "scenes": scenes,
            }
        else:
            scene_manifest = _load_manifest_from_disk()

    if not isinstance(scene_manifest, dict):
        script = str(state.get("script", ""))
        scenes = _parse_script_to_scenes(script)
        scene_manifest = {
            "mode": state.get("mode", "autonomous"),
            "validated": bool(state.get("validated", False)),
            "scene_count": len(scenes),
            "scenes": scenes,
        }

    try:
        task_graph = invoke_tool("get_task_graph", {"scene_manifest": scene_manifest})
        tasks = list(task_graph.get("tasks") or [])
        log_path = str(task_graph.get("log_path") or "")
    except Exception:
        tasks = _fallback_tasks_from_manifest(scene_manifest)
        log_path = ""

    commit_payload = {
        "agent": "scene_parser",
        "scene_count": len(scene_manifest.get("scenes", [])),
        "task_count": len(tasks),
    }
    if log_path:
        commit_payload["task_graph_log"] = log_path

    try:
        invoke_tool("commit_memory", {"data": commit_payload})
    except Exception:
        pass

    updates: Dict[str, Any] = {
        "scene_manifest_data": scene_manifest,
        "scene_tasks": tasks,
    }
    if log_path:
        updates["task_graph_logs"] = [log_path]

    return updates
