from agents.storyboard_artist import (
    _build_decompose_prompt_inputs,
    _format_duration_context,
    system_prompt_template_decompose_visual_description,
    system_prompt_template_design_storyboard,
)
from interfaces import CharacterInScene, ShotBriefDescription


def test_design_prompt_keeps_director_and_continuity_contract() -> None:
    prompt = system_prompt_template_design_storyboard

    assert "Preserve every scripted dialogue line exactly" in prompt
    assert "Reuse the same cam_idx" in prompt
    assert "Track the state of important props across shots" in prompt
    assert "Do not infer role-typical accessories or tools" in prompt


def test_decompose_prompt_contains_bsync_requirements() -> None:
    prompt = system_prompt_template_decompose_visual_description

    assert "concurrent behavior" in prompt
    assert "physical causality" in prompt
    assert "camera movement from actor movement" in prompt
    assert "First Frame -> Motion Process -> Last Frame" in prompt
    assert "listener active" in prompt


def test_decompose_inputs_include_audio_scene_neighbors_and_duration() -> None:
    shot = ShotBriefDescription(
        idx=1,
        is_last=False,
        cam_idx=2,
        visual_desc="<Role A> turns toward <Role B>.",
        audio_desc="[\u8bf4\u8bdd\u4eba] Role A: A short line.",
        estimated_duration_seconds=4.5,
    )
    characters = [
        CharacterInScene(
            idx=0,
            identifier_in_scene="Role A",
            is_visible=True,
            static_features="Generic static features",
            dynamic_features="Generic wardrobe",
        )
    ]

    inputs = _build_decompose_prompt_inputs(
        shot_brief_desc=shot,
        characters=characters,
        scene_context="INT. GENERIC ROOM - DAY",
        previous_shot_context="Shot 0: enters the room.",
        next_shot_context="Shot 2: reacts.",
        duration_seconds=4.5,
        duration_status="director_estimated",
    )

    assert inputs["audio_desc"] == "[\u8bf4\u8bdd\u4eba] Role A: A short line."
    assert inputs["scene_context"] == "INT. GENERIC ROOM - DAY"
    assert inputs["previous_shot_context"] == "Shot 0: enters the room."
    assert inputs["next_shot_context"] == "Shot 2: reacts."
    assert "director estimate" in inputs["duration_context"]
    assert "not a user-specified hard duration" in inputs["duration_context"]


def test_missing_duration_is_not_claimed_as_validated() -> None:
    assert _format_duration_context(None, "not_provided") == "not provided"
