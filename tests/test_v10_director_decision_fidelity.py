import asyncio
import json
from types import SimpleNamespace

from agents.storyboard_artist import (
    StoryboardArtist,
    system_prompt_template_design_storyboard,
)
from interfaces import CharacterInScene, ShotBriefDescription


def _characters() -> list[CharacterInScene]:
    return [
        CharacterInScene(
            idx=0,
            identifier_in_scene="Role A",
            is_visible=True,
            static_features="Generic static features",
            dynamic_features="Generic wardrobe",
        ),
        CharacterInScene(
            idx=1,
            identifier_in_scene="Role B",
            is_visible=True,
            static_features="Generic static features",
            dynamic_features="Generic wardrobe",
        ),
    ]


def test_v10_stage_b_prompt_separates_director_decisions_from_shot_output() -> None:
    prompt = system_prompt_template_design_storyboard

    for phrase in (
        "[Script timeline]",
        "[Beat-to-shot decisions]",
        "[Performance budget]",
        "[Physical and spatial continuity]",
        "Event order is not automatically edit order",
        "Do not impose a minimum shot count",
        "A change in shot size alone is not an editing motivation",
        "Seven or eight seconds of continuous performance must not be placed in a four-second shot",
        "Treat cam_idx as a reusable camera setup",
        "same cam_idx only when camera position, lens, angle, framing, and movement baseline match",
    ):
        assert phrase in prompt

    assert "one line of dialogue per shot" not in prompt.lower()
    assert "shot-reverse-shot is required" not in prompt.lower()


def test_v10_missing_dialogue_triggers_one_targeted_full_board_revision() -> None:
    calls: list[str] = []

    class FakeModel:
        def __or__(self, _parser):
            return self

        async def ainvoke(self, messages, **_kwargs):
            calls.append(json.dumps(messages, ensure_ascii=False, default=str))
            if len(calls) == 1:
                return SimpleNamespace(
                    content=(
                        "Core pressure: Role B withholds a decision while "
                        "Role A tries to force an answer."
                    )
                )
            if len(calls) == 2:
                return SimpleNamespace(
                    storyboard=[
                        ShotBriefDescription(
                            idx=0,
                            is_last=True,
                            cam_idx=0,
                            visual_desc="<Role A> waits for an answer.",
                            audio_desc='[Speaker] Role A: "One."',
                            estimated_duration_seconds=3.0,
                        )
                    ]
                )
            return SimpleNamespace(
                storyboard=[
                    ShotBriefDescription(
                        idx=0,
                        is_last=False,
                        cam_idx=0,
                        visual_desc="<Role A> puts the question to <Role B>.",
                        audio_desc='[Speaker] Role A: "One."',
                        estimated_duration_seconds=2.5,
                    ),
                    ShotBriefDescription(
                        idx=1,
                        is_last=True,
                        cam_idx=1,
                        visual_desc="<Role B> answers without softening the refusal.",
                        audio_desc='[Speaker] Role B: "Two."',
                        estimated_duration_seconds=3.0,
                    ),
                ]
            )

    artist = StoryboardArtist(chat_model=FakeModel())
    result = asyncio.run(
        artist.design_storyboard(
            script='INT. GENERIC ROOM - DAY\nRole A: "One."\nRole B: "Two."',
            characters=_characters(),
            user_requirement="Cinematic, coherent, 16:9",
        )
    )

    assert len(calls) == 3
    assert "scripted dialogue" in calls[2].lower()
    assert artist.last_storyboard_revision_issue
    assert len(artist.last_storyboard_before_revision) == 1
    assert len(artist.last_storyboard_after_revision) == 2
    assert [shot.audio_desc for shot in result] == [
        '[Speaker] Role A: "One."',
        '[Speaker] Role B: "Two."',
    ]
    assert tuple(type(result[0]).model_fields) == (
        "idx",
        "is_last",
        "cam_idx",
        "visual_desc",
        "audio_desc",
        "estimated_duration_seconds",
    )
