from operator import add
from typing import Annotated, Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    mode: str
    input_prompt: str
    manual_script: str
    require_hitl: bool
    approved: bool
    script: str
    validated: bool
    validation_report: Dict[str, Any]
    characters: List[Dict[str, Any]]
    images: List[Dict[str, str]]
    scene_manifest_data: Dict[str, Any]
    scene_tasks: List[Dict[str, Any]]
    scene_payload: Dict[str, Any]
    task_graph_logs: Annotated[List[str], add]
    audio_tracks: Annotated[List[Dict[str, Any]], add]
    video_tracks: Annotated[List[Dict[str, Any]], add]
    face_swaps: List[Dict[str, Any]]
    raw_scenes: List[Dict[str, Any]]
    memory_commit_status: str
    stop_reason: str
    llm_invocations: List[Dict[str, Any]]
    script_repair_count: int