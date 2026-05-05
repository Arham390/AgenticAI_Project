"""Unit tests — Phase 1: Story, Script & Character Design."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.scriptwriter import _fallback_script
from agents.validator import validator_agent
from agents.character import _fallback_characters
from tools.mcp_registry import offline_screenplay_from_prompt


class TestScriptwriter:
    def test_fallback_script_returns_string(self):
        script = _fallback_script("A robot falls in love with the moon.")
        assert isinstance(script, str)
        assert len(script) > 50

    def test_fallback_script_contains_scene_heading(self):
        script = _fallback_script("A detective solves a mystery on Mars.")
        assert "Scene" in script or "INT." in script or "EXT." in script

    def test_fallback_script_contains_dialogue(self):
        script = _fallback_script("Two friends explore a haunted forest.")
        assert ":" in script

    def test_fallback_script_uses_prompt_context(self):
        script = _fallback_script("A story set at the beach at night.")
        lower = script.lower()
        assert "beach" in lower or "night" in lower or "dark" in lower

    def test_offline_screenplay_from_prompt_min_length(self):
        script = offline_screenplay_from_prompt("A quiet coffee shop encounter.")
        assert len(script) > 100

    def test_scriptwriter_agent_with_manual_script(self):
        from agents.scriptwriter import scriptwriter_agent
        state = {
            "input_prompt": "Test prompt",
            "manual_script": "",
            "llm_invocations": [],
        }
        result = scriptwriter_agent(state)
        assert "script" in result
        assert isinstance(result["script"], str)
        assert len(result["script"]) > 0


class TestValidator:
    def _make_state(self, script: str) -> dict:
        return {"script": script, "mode": "manual", "require_hitl": False}

    def test_valid_script_passes(self):
        script = (
            "Scene 1 - EXT. PARK - DAY\n"
            "(Birds chirp in the distance.)\n"
            "ALICE: Good morning, Bob.\n"
            "BOB: Good morning, Alice.\n\n"
            "Scene 2 - INT. CAFE - DAY\n"
            "ALICE: Did you hear the news?\n"
            "BOB: Tell me everything.\n"
        )
        result = validator_agent(self._make_state(script))
        assert result["validated"] is True

    def test_empty_script_fails(self):
        result = validator_agent(self._make_state(""))
        assert result["validated"] is False

    def test_short_script_fails(self):
        result = validator_agent(self._make_state("Hello world"))
        assert result["validated"] is False

    def test_validation_report_is_dict(self):
        result = validator_agent(self._make_state("Some script text here."))
        assert isinstance(result.get("validation_report"), dict)

    def test_multi_scene_script_passes(self):
        script = (
            "Scene 1 - EXT. SPACE - NIGHT\n"
            "(Stars fill the viewport.)\n"
            "ASTRONAUT: The stars are beautiful tonight.\n\n"
            "Scene 2 - INT. SPACESHIP - LATER\n"
            "(The airlock hisses shut.)\n"
            "ASTRONAUT: I must report this discovery.\n"
            "COMMANDER: Understood. Prepare a full report.\n"
        )
        result = validator_agent(self._make_state(script))
        assert result["validated"] is True


class TestCharacterAgent:
    def test_fallback_extracts_characters(self):
        script = (
            "Scene 1 - EXT. FOREST - DAY\n"
            "ALICE: We need to find the path.\n"
            "BOB: I think it's this way.\n"
        )
        chars = _fallback_characters(script)
        assert isinstance(chars, list)
        assert len(chars) >= 1

    def test_fallback_character_has_required_fields(self):
        script = "Scene 1 - INT. ROOM\nLEAD: Let's go!\n"
        chars = _fallback_characters(script)
        for char in chars:
            assert "id" in char
            assert "name" in char

    def test_character_agent_returns_list(self):
        from agents.character import character_agent
        state = {
            "script": "Scene 1\nJESSICA: Hello!\nMARK: Hi there!\n",
            "llm_invocations": [],
        }
        result = character_agent(state)
        assert "characters" in result
        assert isinstance(result["characters"], list)

    def test_no_duplicate_characters(self):
        script = (
            "Scene 1\nALICE: Hello.\nALICE: How are you?\n"
            "Scene 2\nALICE: I am fine.\n"
        )
        chars = _fallback_characters(script)
        names = [c["name"] for c in chars]
        assert len(names) == len(set(names))
