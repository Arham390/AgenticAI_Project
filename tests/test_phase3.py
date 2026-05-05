"""Unit tests — Phase 3: Video Generation & Composition."""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.phase2_media import (
    motion_score_for_frames,
    sync_confidence_for_scene,
    write_scene_metrics,
    apply_face_swap_to_sequence,
    _slugify,
)


class TestFrameGeneration:
    def test_slugify_removes_special_chars(self):
        assert _slugify("hello world!") == "hello_world_"

    def test_slugify_truncates_to_64(self):
        assert len(_slugify("a" * 100)) <= 64

    def test_slugify_empty_returns_asset(self):
        assert _slugify("") == "asset"

    def test_motion_score_empty_dir_is_zero(self):
        with tempfile.TemporaryDirectory() as d:
            score = motion_score_for_frames(Path(d))
            assert score == 0.0

    def test_motion_score_single_frame_is_zero(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            img = Image.new("RGB", (64, 64), (128, 64, 200))
            img.save(Path(d) / "frame_0001.png")
            score = motion_score_for_frames(Path(d))
            assert score == 0.0

    def test_motion_score_identical_frames_near_zero(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            img = Image.new("RGB", (64, 64), (100, 100, 100))
            for i in range(5):
                img.save(Path(d) / f"frame_{i:04d}.png")
            score = motion_score_for_frames(Path(d))
            assert score < 0.01

    def test_sync_confidence_no_audio_is_zero(self):
        with tempfile.TemporaryDirectory() as d:
            conf = sync_confidence_for_scene(Path(d) / "missing.wav", Path(d))
            assert conf == 0.0

    def test_write_scene_metrics_creates_file(self):
        metrics = {"motion_score": 0.05, "identity_confidence": 0.8, "passed": True}
        report_path = write_scene_metrics("test_scene_01", metrics)
        assert Path(report_path).exists()
        data = json.loads(Path(report_path).read_text(encoding="utf-8"))
        assert data["scene_id"] == "test_scene_01"


class TestSceneParser:
    def test_scene_parser_extracts_scenes(self):
        from agents.scene_parser import scene_parser_agent
        script = (
            "Scene 1 - EXT. MARS - DAY\n"
            "ALEX: Amazing view.\n\n"
            "Scene 2 - INT. ROVER - NIGHT\n"
            "ALEX: Time to head back.\n"
        )
        state = {"script": script, "mode": "manual", "validated": True,
                 "scene_manifest_data": {}, "scene_tasks": [], "task_graph_logs": []}
        result = scene_parser_agent(state)
        manifest = result.get("scene_manifest_data", {})
        assert isinstance(manifest.get("scenes"), list)
        assert len(manifest["scenes"]) >= 2

    def test_scene_parser_assigns_scene_ids(self):
        from agents.scene_parser import scene_parser_agent
        script = "Scene 1 - EXT. PARK - DAY\nLEAD: Hello world.\n"
        state = {"script": script, "mode": "manual", "validated": True,
                 "scene_manifest_data": {}, "scene_tasks": [], "task_graph_logs": []}
        result = scene_parser_agent(state)
        scenes = result["scene_manifest_data"].get("scenes", [])
        for scene in scenes:
            assert "scene_id" in scene

    def test_scene_parser_extracts_dialogues(self):
        from agents.scene_parser import scene_parser_agent
        script = "Scene 1\nALICE: First line.\nBOB: Second line.\n"
        state = {"script": script, "mode": "manual", "validated": True,
                 "scene_manifest_data": {}, "scene_tasks": [], "task_graph_logs": []}
        result = scene_parser_agent(state)
        scenes = result["scene_manifest_data"]["scenes"]
        assert any(len(s.get("dialogues", [])) > 0 for s in scenes)


class TestVideoGenAgent:
    def _state(self):
        return {
            "scene_manifest_data": {
                "scenes": [
                    {"scene_id": "scene_01", "heading": "Scene 1 - EXT. SPACE", "dialogues": [
                        {"character": "Alex", "line": "Hello!"}
                    ], "actions": []}
                ]
            },
            "images": [],
            "video_tracks": [],
        }

    def test_video_gen_agent_returns_video_tracks(self):
        import os
        os.environ.setdefault("PHASE2_STRICT_PRODUCTION", "0")
        from agents.video_gen import video_gen_agent
        result = video_gen_agent(self._state())
        assert "video_tracks" in result
        assert isinstance(result["video_tracks"], list)

    def test_video_gen_has_scene_id(self):
        import os
        os.environ.setdefault("PHASE2_STRICT_PRODUCTION", "0")
        from agents.video_gen import video_gen_agent
        result = video_gen_agent(self._state())
        for track in result["video_tracks"]:
            assert "scene_id" in track
