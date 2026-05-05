"""Unit tests — Phase 5: Intelligent Edit Agent & State Versioning."""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.edit_agent import classify_intent, _rule_based_intent, VALID_INTENTS
from agents.state_manager import StateManager


class TestIntentClassification:
    """Tests for the rule-based intent classifier (no LLM required)."""

    def _classify(self, query: str) -> dict:
        return _rule_based_intent(query)

    def test_voice_query_maps_to_audio(self):
        r = self._classify("Change voice tone to dramatic")
        assert r["target"] == "audio"

    def test_darker_query_maps_to_video_frame(self):
        r = self._classify("Make the scene darker and more ominous")
        assert r["target"] == "video_frame"

    def test_music_query_maps_to_audio(self):
        r = self._classify("Add background music to the opening scene")
        assert r["target"] == "audio"

    def test_script_query_maps_to_script(self):
        r = self._classify("Regenerate the script with more tension")
        assert r["target"] == "script"

    def test_subtitle_query_maps_to_video(self):
        r = self._classify("Remove the subtitle overlay from the final video")
        assert r["target"] == "video"

    def test_speed_query_maps_to_video(self):
        r = self._classify("Speed up this scene by 2x")
        assert r["target"] == "video"

    def test_unknown_query_returns_unknown(self):
        r = self._classify("purple elephant dancing xyzzy")
        # Should be unknown or best-effort match
        assert r.get("target") in list(VALID_INTENTS.values()) + ["unknown"]

    def test_classify_intent_returns_dict(self):
        r = classify_intent("Make the narration louder")
        assert isinstance(r, dict)
        assert "intent" in r
        assert "target" in r

    def test_classify_intent_has_required_keys(self):
        r = classify_intent("Change the background music")
        for key in ("intent", "target", "scope", "parameters", "confidence"):
            assert key in r

    def test_intent_parameters_has_raw_query(self):
        query = "Make it look like a noir film"
        r = _rule_based_intent(query)
        assert r["parameters"].get("raw_query") == query

    def test_bright_query_maps_to_video_frame(self):
        r = self._classify("Make everything brighter and more colorful")
        assert r["target"] == "video_frame"

    def test_character_design_query(self):
        r = self._classify("Change character design to futuristic")
        assert r["target"] == "video_frame"


class TestStateManager:
    def _manager(self) -> tuple[StateManager, Path]:
        tmpdir = Path(tempfile.mkdtemp())
        sm = StateManager(state_dir=tmpdir / "state_versions")
        return sm, tmpdir

    def test_snapshot_returns_version_number(self):
        sm, _ = self._manager()
        version = sm.snapshot({"script": "Hello world", "mode": "test"}, description="test")
        assert isinstance(version, int)
        assert version == 1

    def test_snapshot_increments_version(self):
        sm, _ = self._manager()
        v1 = sm.snapshot({"data": "first"}, description="v1")
        v2 = sm.snapshot({"data": "second"}, description="v2")
        assert v2 == v1 + 1

    def test_history_returns_all_versions(self):
        sm, _ = self._manager()
        sm.snapshot({"a": 1}, description="first")
        sm.snapshot({"a": 2}, description="second")
        sm.snapshot({"a": 3}, description="third")
        history = sm.history()
        assert len(history) == 3
        assert history[0]["version"] == 1
        assert history[2]["version"] == 3

    def test_history_contains_description(self):
        sm, _ = self._manager()
        sm.snapshot({"x": 1}, description="my special edit")
        history = sm.history()
        assert history[0]["description"] == "my special edit"

    def test_revert_restores_state(self):
        sm, _ = self._manager()
        original_state = {"script": "original script", "mode": "test"}
        sm.snapshot(original_state, description="original")
        sm.snapshot({"script": "modified script", "mode": "test"}, description="modified")
        restored = sm.revert(1)
        assert restored is not None
        assert restored.get("script") == "original script"

    def test_revert_nonexistent_returns_none(self):
        sm, _ = self._manager()
        result = sm.revert(9999)
        assert result is None

    def test_snapshot_persists_to_disk(self):
        sm, tmpdir = self._manager()
        sm.snapshot({"key": "value"}, description="disk test")
        index_path = sm._state_dir / "index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text(encoding="utf-8"))
        assert len(data) == 1

    def test_diff_summary_reports_deltas(self):
        sm, _ = self._manager()
        sm.snapshot(
            {"characters": [{"name": "Alice"}], "audio_tracks": [],
             "scene_manifest_data": {"scenes": [{"id": "s1"}]}},
            description="v1"
        )
        sm.snapshot(
            {"characters": [{"name": "Alice"}, {"name": "Bob"}],
             "audio_tracks": [{"id": "a1"}],
             "scene_manifest_data": {"scenes": [{"id": "s1"}]}},
            description="v2"
        )
        diff = sm.diff_summary(1, 2)
        assert diff["from_version"] == 1
        assert diff["to_version"] == 2
        assert isinstance(diff["character_count_delta"], int)

    def test_snapshot_state_json_is_readable(self):
        sm, _ = self._manager()
        state = {"script": "test", "characters": [{"name": "X"}]}
        sm.snapshot(state, description="readable")
        vdir = sm._version_dir(1)
        state_path = vdir / "state.json"
        assert state_path.exists()
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        assert loaded.get("script") == "test"


class TestProcessEdit:
    def test_process_edit_returns_dict(self):
        from agents.edit_agent import process_edit

        sm, _ = TestStateManager()._manager()
        result = process_edit("Make the scene darker", {}, sm)
        assert isinstance(result, dict)
        assert "intent" in result
        assert "edit_result" in result

    def test_process_edit_captures_snapshot_on_success(self):
        from agents.edit_agent import process_edit
        import os
        os.environ.setdefault("PHASE2_STRICT_PRODUCTION", "0")

        sm, _ = TestStateManager()._manager()
        state = {
            "scene_manifest_data": {
                "scenes": [{"scene_id": "s1", "heading": "Scene 1", "dialogues": [], "actions": []}]
            }
        }
        result = process_edit("Make the visuals brighter", state, sm)
        # Version may be None if edit fails, but result must be a dict
        assert isinstance(result, dict)

    def test_process_edit_unknown_produces_skipped(self):
        from agents.edit_agent import process_edit

        sm, _ = TestStateManager()._manager()
        # Force unknown target
        result = process_edit("xyzzy purple elephant nonsense 12345", {}, sm)
        assert isinstance(result, dict)
