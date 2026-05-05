"""Unit tests — Phase 2: Audio Generation & TTS."""
import sys
import wave
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.phase2_media import (
    synthesize_voice_wav,
    _fallback_tone_wav,
    _safe_text,
    _audio_duration_seconds,
)


class TestVoiceSynthesis:
    def _tmp_wav(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        return Path(f.name)

    def test_synthesize_returns_bool_and_backend(self):
        p = self._tmp_wav()
        ok, backend = synthesize_voice_wav(p, "Hello world.", "TestChar", "neutral")
        assert isinstance(ok, bool)
        assert isinstance(backend, str)
        p.unlink(missing_ok=True)

    def test_synthesize_creates_wav_file(self):
        p = self._tmp_wav()
        ok, _ = synthesize_voice_wav(p, "Testing audio synthesis.", "Alice", "neutral")
        assert ok is True
        assert p.exists()
        assert p.stat().st_size > 0
        p.unlink(missing_ok=True)

    def test_fallback_tone_wav_produces_valid_wav(self):
        p = self._tmp_wav()
        _fallback_tone_wav(p, "Hello there.", "Bob", "neutral")
        assert p.exists()
        with wave.open(str(p), "rb") as wf:
            assert wf.getnframes() > 0
            assert wf.getframerate() > 0
        p.unlink(missing_ok=True)

    def test_fallback_tone_duration_scales_with_text(self):
        short_p = self._tmp_wav()
        long_p = self._tmp_wav()
        _fallback_tone_wav(short_p, "Hi.", "X", "neutral")
        _fallback_tone_wav(long_p, "This is a much longer sentence that should produce a longer audio file.", "X", "neutral")
        assert long_p.stat().st_size >= short_p.stat().st_size
        short_p.unlink(missing_ok=True)
        long_p.unlink(missing_ok=True)

    def test_fallback_tone_wav_different_emotions(self):
        for emotion in ("neutral", "sad", "happy", "angry", "calm"):
            p = self._tmp_wav()
            _fallback_tone_wav(p, "Testing emotion.", "Char", emotion)
            assert p.exists()
            p.unlink(missing_ok=True)

    def test_safe_text_returns_placeholder_on_empty(self):
        result = _safe_text("")
        assert len(result) > 0

    def test_safe_text_strips_whitespace(self):
        result = _safe_text("  hello  ")
        assert result == "hello"

    def test_audio_duration_seconds_on_valid_wav(self):
        p = self._tmp_wav()
        _fallback_tone_wav(p, "Hello.", "X", "neutral")
        dur = _audio_duration_seconds(p)
        assert dur > 0.0
        p.unlink(missing_ok=True)

    def test_audio_duration_seconds_on_missing_file(self):
        dur = _audio_duration_seconds(Path("/nonexistent/path/file.wav"))
        assert dur == 0.0


class TestVoiceSynthAgent:
    def _scene_state(self, dialogues=None):
        if dialogues is None:
            dialogues = [{"character": "Alice", "line": "Hello there!"}]
        return {
            "scene_manifest_data": {
                "scenes": [
                    {"scene_id": "scene_01", "heading": "Scene 1", "dialogues": dialogues, "actions": []}
                ]
            },
            "audio_tracks": [],
        }

    def test_voice_synth_agent_returns_audio_tracks(self):
        from agents.voice_synth import voice_synth_agent
        state = self._scene_state()
        result = voice_synth_agent(state)
        assert "audio_tracks" in result
        assert isinstance(result["audio_tracks"], list)
        assert len(result["audio_tracks"]) > 0

    def test_voice_synth_agent_has_scene_id(self):
        from agents.voice_synth import voice_synth_agent
        state = self._scene_state()
        result = voice_synth_agent(state)
        for track in result["audio_tracks"]:
            assert "scene_id" in track

    def test_voice_synth_agent_handles_empty_scenes(self):
        from agents.voice_synth import voice_synth_agent
        state = {"scene_manifest_data": {"scenes": []}, "audio_tracks": []}
        result = voice_synth_agent(state)
        assert result["audio_tracks"] == []

    def test_voice_synth_agent_multiple_dialogues(self):
        from agents.voice_synth import voice_synth_agent
        state = self._scene_state([
            {"character": "Alice", "line": "First line."},
            {"character": "Bob", "line": "Second line."},
        ])
        result = voice_synth_agent(state)
        track = result["audio_tracks"][0]
        assert len(track.get("clips", [])) == 2
