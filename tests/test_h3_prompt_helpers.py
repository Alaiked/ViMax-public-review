from scripts.h3_production_agent import (
    _append_missing_dialogues,
    _append_shot_soundscape,
    _extract_dialogues,
    _split_prompt_by_shot,
)


def test_dialogue_extraction_preserves_audio_order() -> None:
    dialogues = _extract_dialogues(
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cFirst line.\u201d\n"
        "[\u8bf4\u8bdd\u4eba] Role B: \u201cSecond line.\u201d",
        {},
    )

    assert [item["speaker_id"] for item in dialogues] == ["Role A", "Role B"]
    assert [item["text"] for item in dialogues] == ["First line.", "Second line."]


def test_missing_dialogues_are_appended_in_source_order() -> None:
    prompt = _append_missing_dialogues(
        "Base prompt.",
        [
            {"speaker_id": "char_1", "text": "First line."},
            {"speaker_id": "char_2", "text": "Second line."},
        ],
    )

    assert prompt.index("First line.") < prompt.index("Second line.")
    assert "<Subject 1>" in prompt
    assert "<Subject 2>" in prompt


def test_soundscape_keeps_dialogue_out_of_ambient_tail() -> None:
    prompt = _append_shot_soundscape(
        "Shot body.",
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cLine.\u201d\n[Sound Effect] Room tone.",
    )

    assert "overall_soundscape:" in prompt
    assert "Room tone." in prompt
    assert "Role A" not in prompt


def test_prompt_split_keeps_common_preamble_and_shot_sections() -> None:
    prompt = (
        "Common header\n\n"
        "[Shot 0]\nFirst shot.\n\n"
        "[Shot 1]\nSecond shot.\n\n"
        "overall_soundscape:\nRoom tone."
    )

    preamble, sections = _split_prompt_by_shot(prompt, 2)

    assert preamble == "Common header"
    assert len(sections) == 2
    assert "[Shot 0]" in sections[0]
    assert "[Shot 1]" in sections[1]
    assert "overall_soundscape:" not in sections[1]
