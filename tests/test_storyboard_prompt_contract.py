import asyncio
import json
from types import SimpleNamespace

from agents.storyboard_artist import (
    StoryboardArtist,
    _build_decompose_prompt_inputs,
    _format_duration_context,
    human_prompt_template_design_storyboard,
    human_prompt_template_interpret_scene,
    system_prompt_template_decompose_visual_description,
    system_prompt_template_design_storyboard,
    system_prompt_template_interpret_scene,
)
from interfaces import CharacterInScene, ShotBriefDescription


def test_stage_a_and_stage_b_keep_separate_director_responsibilities() -> None:
    interpretation_prompt = system_prompt_template_interpret_scene
    design_prompt = system_prompt_template_design_storyboard

    for phrase in (
        "central dramatic pressure",
        "playable beats",
        "Do not force a fixed list of beat labels",
        "Information and attention",
        "objective, tactic, obstacle, and change of tactic",
        "editing motivations",
        "final residue",
    ):
        assert phrase in interpretation_prompt

    for phrase in (
        "original script and supplied character definitions are the final authority",
        "director's interpretation is advisory creative guidance",
        "Preserve every scripted dialogue line exactly once",
        "Organize the board by dramatic beats",
        "not by dialogue count, action-step count",
        "primary dramatic function",
        "A change in shot size alone is not an editing motivation",
        "Treat cam_idx as a reusable camera setup",
        "Do not copy source time windows",
    ):
        assert phrase in design_prompt

    assert "{script_str}" in human_prompt_template_interpret_scene
    assert "{characters_str}" in human_prompt_template_interpret_scene
    assert "<DIRECTOR_INTERPRETATION>" in human_prompt_template_design_storyboard
    assert "{director_interpretation}" in human_prompt_template_design_storyboard


def test_design_storyboard_runs_interpretation_before_structured_design() -> None:
    calls: list[str] = []

    class FakeModel:
        def __or__(self, _parser):
            return self

        async def ainvoke(self, messages, **_kwargs):
            calls.append(json.dumps(messages, ensure_ascii=False, default=str))
            if len(calls) == 1:
                return SimpleNamespace(
                    content=(
                        "Directing brief: the scene tests whether Role A can "
                        "trust Role B. The final residue is a withheld answer."
                    )
                )
            return SimpleNamespace(
                storyboard=[
                    ShotBriefDescription(
                        idx=0,
                        is_last=True,
                        cam_idx=0,
                        visual_desc="<Role A> watches <Role B> across a table.",
                        audio_desc="[Speaker] Role A: You knew.",
                        estimated_duration_seconds=3.5,
                    )
                ]
            )

    artist = StoryboardArtist(chat_model=FakeModel())
    result = asyncio.run(
        artist.design_storyboard(
            script="INT. GENERIC ROOM - DAY\nROLE A: You knew.",
            characters=[
                CharacterInScene(
                    idx=0,
                    identifier_in_scene="Role A",
                    is_visible=True,
                    static_features="Generic static features",
                    dynamic_features="Generic wardrobe",
                )
            ],
            user_requirement="Cinematic, coherent, 16:9",
        )
    )

    assert len(calls) == 2
    assert "central dramatic pressure" in calls[0].lower()
    assert "Directing brief: the scene tests whether Role A can trust Role B." in calls[1]
    assert "INT. GENERIC ROOM - DAY" in calls[1]
    assert "Character 0" in calls[1]
    assert artist.last_director_interpretation.startswith("Directing brief:")
    assert tuple(type(result[0]).model_fields) == (
        "idx",
        "is_last",
        "cam_idx",
        "visual_desc",
        "audio_desc",
        "estimated_duration_seconds",
    )


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
