from typing import Any, Dict, List

from tools.mcp_registry import invoke_tool


def _iter_target_scenes(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    scene_payload = state.get("scene_payload")
    if isinstance(scene_payload, dict):
        return [scene_payload]

    manifest = state.get("scene_manifest_data")
    if isinstance(manifest, dict) and isinstance(manifest.get("scenes"), list):
        return list(manifest.get("scenes") or [])

    return []


def _scene_voice_payload(scene: Dict[str, Any], idx: int) -> Dict[str, Any]:
    scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
    heading = str(scene.get("heading", f"Scene {idx}"))

    dialogues = scene.get("dialogues", [])
    if not isinstance(dialogues, list):
        dialogues = []

    if not dialogues:
        dialogues = [
            {
                "character": "Narrator",
                "line": f"{heading}. Silent beat for visual continuity.",
            }
        ]

    clips: List[Dict[str, Any]] = []
    for clip_index, dialogue in enumerate(dialogues, start=1):
        character = str(dialogue.get("character", "Narrator"))
        line = str(dialogue.get("line", "")) or f"Narration for {heading}."
        audio_path = invoke_tool(
            "voice_cloning_synthesizer",
            {
                "scene_id": scene_id,
                "character_name": character,
                "text": line,
                "emotion": "neutral",
            },
        )
        clips.append(
            {
                "clip_id": f"{scene_id}_clip_{clip_index:02d}",
                "character": character,
                "line": line,
                "audio_path": audio_path,
            }
        )

    primary_audio = clips[0]["audio_path"] if clips else ""
    return {
        "scene_id": scene_id,
        "primary_audio": primary_audio,
        "clips": clips,
    }


def voice_synth_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    scenes = _iter_target_scenes(state)
    outputs: List[Dict[str, Any]] = []

    for idx, scene in enumerate(scenes, start=1):
        try:
            outputs.append(_scene_voice_payload(scene, idx))
        except Exception as exc:
            scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
            outputs.append(
                {
                    "scene_id": scene_id,
                    "primary_audio": "",
                    "clips": [],
                    "error": str(exc),
                }
            )

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "voice_synth",
                    "scene_audio_count": len(outputs),
                }
            },
        )
    except Exception:
        pass

    return {"audio_tracks": outputs}
