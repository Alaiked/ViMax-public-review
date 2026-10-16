import pytest

from agents.storyboard_artist import validate_dialogue_performance


def test_dialogue_order_regression_is_rejected_and_correct_order_passes() -> None:
    audio_desc = (
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cFirst line.\u201d\n"
        "[\u8bf4\u8bdd\u4eba] Role B: \u201cSecond line.\u201d"
    )
    v8_style_motion_desc = (
        "Role B says \u201cSecond line.\u201d "
        "Role A then says \u201cFirst line.\u201d"
    )
    corrected_motion_desc = (
        "Role A says \u201cFirst line.\u201d "
        "Role B then says \u201cSecond line.\u201d"
    )

    with pytest.raises(ValueError, match="order"):
        validate_dialogue_performance(audio_desc, v8_style_motion_desc)

    validate_dialogue_performance(audio_desc, corrected_motion_desc)


def test_dialogue_omission_is_rejected() -> None:
    audio_desc = (
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cFirst line.\u201d\n"
        "[\u8bf4\u8bdd\u4eba] Role B: \u201cSecond line.\u201d"
    )

    with pytest.raises(ValueError, match="omitted"):
        validate_dialogue_performance(audio_desc, "Role A speaks only once.")


def test_repeated_dialogue_requires_each_occurrence_in_sequence() -> None:
    audio_desc = (
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cWait.\u201d\n"
        "[\u8bf4\u8bdd\u4eba] Role B: \u201cNo.\u201d\n"
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cWait.\u201d"
    )
    incomplete_motion_desc = (
        "Role A says \u201cWait.\u201d "
        "Role B then says \u201cNo.\u201d"
    )
    corrected_motion_desc = (
        "Role A says \u201cWait.\u201d "
        "Role B then says \u201cNo.\u201d "
        "Role A finally says \u201cWait.\u201d"
    )

    with pytest.raises(ValueError, match="order|multiplicity|omitted"):
        validate_dialogue_performance(audio_desc, incomplete_motion_desc)

    validate_dialogue_performance(audio_desc, corrected_motion_desc)


def test_explicit_wrong_speaker_is_rejected() -> None:
    audio_desc = (
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cOne.\u201d\n"
        "[\u8bf4\u8bdd\u4eba] Role B: \u201cTwo.\u201d"
    )
    wrong_speaker_motion_desc = (
        "Role B says \u201cOne.\u201d "
        "Role A then says \u201cTwo.\u201d"
    )

    with pytest.raises(ValueError, match="speaker"):
        validate_dialogue_performance(audio_desc, wrong_speaker_motion_desc)
