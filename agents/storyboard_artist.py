from typing import List, NamedTuple, Optional, Literal
import asyncio
import re
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from interfaces import CharacterInScene, ShotDescription, ShotBriefDescription

from utils.retry import after_func



system_prompt_template_interpret_scene = \
"""
[Role]
You are the director preparing one scene before any shots are chosen. You are not writing coverage, a plot summary, or a new screenplay.

[Task]
Read the complete scene and the supplied character definitions. Build a compact directing brief that identifies what the scene is really doing and how the audience should experience it. The script remains the final authority: do not add, remove, reorder, reinterpret, or rewrite events, dialogue, props, space, or character motivation.

[Decide]
- The scene's central dramatic pressure: what changes, who is trying to make it change, who resists, and what each person cannot openly say.
- The playable beats: only the turns that change knowledge, power, intimacy, risk, or intention. Name a beat by its dramatic event, not by a camera move. Do not force a fixed list of beat labels, and do not equate one dialogue line or one action step with one beat.
- Information and attention: what the audience should know at each turn, when to watch the person acting, the person receiving, a revealing object, or the space between them, and what should be withheld.
- Character-specific play: for each participant, a concrete objective, tactic, obstacle, and change of tactic in this scene. Explain why the same line or action would be performed differently by this character.
- Visual and rhythmic organization: the meaningful spatial relationships, visual center, axis, major reaction or silence, pacing shifts, and possible editing motivations. Choose precise visual ideas, but do not prescribe a fixed shot template.
- The final residue: what image, pause, or unanswered pressure should remain after the last scripted event.

[How to write]
- Write in the language of the script.
- Be concrete and selective. Use short labeled paragraphs only where they clarify the choice; omit a category if it does not matter for this scene. Never fill categories mechanically.
- Distinguish a beat from an action step. A beat may contain several lines, simultaneous behavior, a silence, or one decisive gesture.
- Treat personality notes as evidence for behavior, not as labels.
- Keep the brief compact enough to guide a separate storyboard call. Do not list every line of dialogue.
- Do not propose extra characters, props, locations, events, backstory, or motivations. If the script is silent on a fact, do not invent it.
"""


human_prompt_template_interpret_scene = \
"""
<SCRIPT>
{script_str}
</SCRIPT>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>
"""


system_prompt_template_design_storyboard = \
"""
[Role]
You are a professional film director and storyboard artist. Convert the original scene and the director's interpretation into a complete production-ready storyboard using the supplied structured schema.

[Authority]
- The original script and supplied character definitions are the final authority for story events, dialogue, props, space, and motivation.
- The director's interpretation is advisory creative guidance. It must not override or add to the script.
- Preserve every scripted dialogue line exactly once, in the original order, in the audio description of the shots. Do not summarize, paraphrase, merge, or invent lines.
- Do not add characters, extras, locations, props, or events that are not supported by the script.

[Shot design]
- Organize the board by dramatic beats, not by dialogue count, action-step count, or a mechanical shot-reverse-shot pattern.
- Each shot must have one primary dramatic function: reveal information, change power or intimacy, stage a decision, register a reaction, establish a necessary spatial relation, or create a deliberate hold or ellipsis.
- Choose what the audience first sees, where attention moves, and what image or tension remains at the end of the shot. A shot may contain multiple lines or actions when they form one beat.
- Cut when attention, knowledge, power, space, point of view, or rhythm changes. A change in shot size alone is not an editing motivation.
- Use reactions, off-screen presence, occlusion, foreground layering, spatial separation, action matching, or stillness when they carry meaning. Do not force any device or movement into every shot.
- Convert character objectives and tactics into concrete playable behavior: listening, withholding, testing, deciding, resisting, committing, changing tactics, or reacting at a specific moment. Avoid generic expression labels and decorative gestures.
- Keep a decisive cause and its visible consequence together when they form one continuous action. Do not split preparation, execution, contact, and result into routine prop inserts unless isolation or withheld information has a real dramatic purpose.
- Describe the camera precisely: shot size, angle, composition, focus, movement, and the reason the frame changes. Camera movement must reveal, follow, pressure, isolate, connect, or withhold; static framing is valid when the beat needs it.
- Maintain a consistent 180-degree action axis, eyeline, screen direction, spatial geography, and prop state across cuts. Motivate a necessary axis change visibly.
- Make important actions, contact, and prop state visible within the intended composition. Do not assert a scripted action off-screen when the beat depends on seeing it.
- Treat cam_idx as a reusable camera setup, not a shot number. Reuse a setup only when position, lens, angle, framing, and movement baseline genuinely match.
- Estimate estimated_duration_seconds from the full performance: dialogue length, pauses, concurrent behavior, physical travel, contact, and the final reaction. Do not copy source time windows or mechanically multiply action steps. Mark it honestly as a director estimate.

[Visible storyboard contract]
- Every visual description must be self-contained and use frame-relative placement and clear direction of gaze and movement.
- Keep character names in angle brackets inside visual descriptions, but not in speaker fields or dialogue.
- Make the sequence continuous in character, space, and important-prop state without forcing the reader to invent transitions.
- When a close shot focuses on a person, specify the visible body part and the performance information it carries.
- Keep interior/exterior, time of day, and location consistent with the script.

[Input]
The user provides the original script, character definitions, and an optional requirement.

[Output]
{format_instructions}
"""


human_prompt_template_design_storyboard = \
"""
<SCRIPT>
{script_str}
</SCRIPT>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>

<DIRECTOR_INTERPRETATION>
{director_interpretation}
</DIRECTOR_INTERPRETATION>
"""


def _message_content_to_text(content) -> str:
    """Normalize plain-text and content-block chat responses."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content or "").strip()



system_prompt_template_decompose_visual_description = \
"""
[Role]
You are a professional visual text analyst and B-Sync motion writer, proficient in cinematic language, physical performance, and shot narration. Your expertise lies in deconstructing a comprehensive shot description accurately into three core components: the static first frame, the static last frame, and the dynamic motion process that connects them.

[Task]
Your task is to dissect and rewrite a user-provided visual text description of a shot strictly and insightfully into three distinct parts:
- First Frame Description: Describe the static image at the very beginning of the shot. Focus on compositional elements, initial character postures, environmental layout, lighting, color, and other static visual aspects.
- Last Frame Description: Describe the static image at the very end of the shot. Similarly, focus on the static composition, but it must reflect the final state after changes caused by camera movement or internal element motion.
- Motion Description: Describe all movements that occur between the first frame and the last frame. This includes camera movement (e.g., static, push-in, pull-out, pan, track, follow, tilt, etc.) and movement of elements within the shot (e.g., character movement, object displacement, changes in lighting, etc.). This is the most dynamic part of the entire description. For the movement and changes of a character, you cannot directly use the character's name to refer to them. Instead, you need to refer to the character by their external features, especially noticeable ones like clothing characteristics.

[Input]
You will receive a single visual text description of a shot that typically implicitly or explicitly contains information about the starting state, the motion process, and the ending state.
Additionally, you will receive a sequence of potential characters, each containing an identifier and a feature.
- The description is enclosed within <VISUAL_DESC> and </VISUAL_DESC>.
- The character list is enclosed within <CHARACTERS> and </CHARACTERS>.
- The shot audio description is enclosed within <AUDIO_DESC> and </AUDIO_DESC>.
- A concise scene context is enclosed within <SCENE_CONTEXT> and </SCENE_CONTEXT>.
- The immediately previous and next shot summaries are enclosed within <PREVIOUS_SHOT_CONTEXT> and <NEXT_SHOT_CONTEXT>.
- The intended performance duration, when supplied, is enclosed within <DURATION_CONTEXT>. If it says "not provided", no duration was supplied.


[Output]
{format_instructions}

[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Ensure the first and last frame descriptions are pure "snapshots," containing no ongoing actions (e.g., "He is about to stand up" is unacceptable; it should be "He is sitting on the chair, leaning slightly forward").
- In the motion description, you must clearly distinguish between camera movement and on-screen movement. Use professional cinematic terminology (e.g., dolly shot, pan, zoom, etc.) as precisely as possible to describe camera movement.
- In the motion description, you cannot directly use character names to refer to characters; instead, you should use the characters' visible characteristics to refer to them. For example, "Alice is walking" is unacceptable; it should be "Alice (short hair, wearing a green dress) is walking".
- The last frame description must be logically consistent with the first frame description and the motion description. All actions described in the motion section should be reflected in the static image of the last frame.
- Write the motion as a clear temporal sequence with explicit action timing. State the initiating action, the intermediate development, and the resulting end state; avoid simultaneous-sounding clauses when the actions must occur one after another.
- When multiple actors move at the same time, describe their concurrent behavior as coordinated parallel actions and clarify which action happens first, which overlaps, and which completes last.
- Specify actual physical contact when the source action implies it (for example, hand meeting ball, foot planting on the floor, shoulder brushing past another actor), and describe the physical causality that produces the next state. For weight-bearing contact such as sitting, lifting, pulling, placing, or releasing, write the approach, first contact, pressure or resistance, settle, and stabilized aftermath as one causal sequence.
- Preserve exact prop and setting identities from the visual description. Do not replace period-specific nouns or alter the stated object. If the shot depends on a hand, foot, or object contact, keep that contact visibly inside the specified composition instead of merely asserting that it happened off-screen.
- Clearly distinguish camera movement from actor movement. Name the camera movement separately, then describe actor movement in relation to the moving frame.
- Maintain a complete First Frame -> Motion Process -> Last Frame state closure: the motion must begin from the stated first-frame state and end in the stated last-frame state without unexplained teleportation, pose jumps, or contradictory contact.
- When audio_desc contains a [Speaker] line, coordinate the visible performance with it: use natural speaking lip movement, breath, vocal effort, gaze, and interaction when the speaker is visible. If the speaker is not visible in the frame, keep the listener active with a timed reaction instead of making the listener motionless.
- Preserve every scripted dialogue line from audio_desc verbatim in motion_desc and perform the lines in their original order. Do not omit, paraphrase, merge, or reorder dialogue while writing the motion process.
- Use scene_context for environment and continuity only. Use previous_shot_context and next_shot_context to preserve necessary start and end states, while keeping the current shot self-contained. Do not copy the whole scene or invent events not supported by the context.
- Adapt the amount of performance to the intended performance duration when DURATION_CONTEXT provides one. Estimate the whole shot, not by mechanically multiplying the number of action steps. A source time window or a phrase such as "the last approximately 0.6 seconds" is not automatically the duration of the whole shot. Respect the duration source: a user-provided duration is a hard constraint, while a director estimate is a planning estimate and not a user-specified hard duration. Focus on actions and reactions that can physically be performed in that time; compress or omit subordinate business rather than overloading the shot. Do not mechanically add a time window to every action.
- If DURATION_CONTEXT says "not provided", plan coherent, plausible motion from the available visual evidence but do not claim that timing validation passed, and do not fabricate a duration.
- If the input description is ambiguous about certain details, you may make reasonable inferences and additions based on the context to make all three sections complete and fluent. However, core elements must strictly adhere to the input text.
- Use accurate, concise, and professional descriptive language. Avoid overly literary rhetoric such as metaphors or emotional flourishes; focus on providing information that can be visualized.
- Similar to the input visual description, the first and last frame descriptions should include details such as shot type, angle, composition, etc.
- Below are the three types of variation within a shot (not between two shots):
(1) 'large' cases typically involve the exaggerated transition shots which means a significant change in the composition and focus, such as smoothly changing from a wide shot to a close-up. It is usually accompanied by significant camera movement (e.g., drone perspective shots across the city).
(2) 'medium' cases often involve the introduction of new characters and a character turns from the back to face the front (facing the camera).
(3) 'small' cases usually involve minor changes, such as expression changes, movement and pose changes of existing characters(e.g., walking, sitting down, standing up), moderate camera movements(e.g., pan, tilt, track).
- When describing a character, it is necessary to indicate the direction they are facing.
"""


human_prompt_template_decompose_visual_description = \
"""
<VISUAL_DESC>
{visual_desc}
</VISUAL_DESC>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<AUDIO_DESC>
{audio_desc}
</AUDIO_DESC>

<SCENE_CONTEXT>
{scene_context}
</SCENE_CONTEXT>

<PREVIOUS_SHOT_CONTEXT>
{previous_shot_context}
</PREVIOUS_SHOT_CONTEXT>

<NEXT_SHOT_CONTEXT>
{next_shot_context}
</NEXT_SHOT_CONTEXT>

<DURATION_CONTEXT>
{duration_context}
</DURATION_CONTEXT>
"""


def _format_duration_context(
    duration_seconds: Optional[float],
    duration_status: Optional[str] = None,
) -> str:
    if duration_seconds is None:
        return "not provided"
    value = float(duration_seconds)
    if value.is_integer():
        value = int(value)
    if duration_status == "director_estimated":
        return (
            f"{value} seconds (director estimate; not a user-specified hard duration)"
        )
    if duration_status == "provided":
        return f"{value} seconds (user-provided hard duration)"
    return f"{value} seconds"


def _build_decompose_prompt_inputs(
    shot_brief_desc: ShotBriefDescription,
    characters: List[CharacterInScene],
    scene_context: str = "",
    previous_shot_context: str = "",
    next_shot_context: str = "",
    duration_seconds: Optional[float] = None,
    duration_status: Optional[str] = None,
):
    characters_str = "\n".join(
        [
            f"{char.identifier_in_scene}: (static) {char.static_features}; (dynamic) {char.dynamic_features}"
            for char in characters
        ]
    )
    return {
        "visual_desc": shot_brief_desc.visual_desc.strip(),
        "characters_str": characters_str,
        "audio_desc": shot_brief_desc.audio_desc.strip() or "none",
        "scene_context": scene_context.strip() or "not provided",
        "previous_shot_context": previous_shot_context.strip() or "not provided",
        "next_shot_context": next_shot_context.strip() or "not provided",
        "duration_context": _format_duration_context(
            duration_seconds,
            duration_status,
        ),
    }


_AUDIO_DIALOGUE_RE = re.compile(
    r"(?P<speaker>[^:\uff1a\n]{1,120}?)[:\uff1a]\s*"
    r"[\u201c\"](?P<text>[^\u201d\"]+)[\u201d\"]"
)
_MARKERLESS_SPEAKER_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z][^:\uff1a\n]{0,80}?"
    r"[\uff08(][^:\uff1a\n]{0,80}?[\uff09)][^:\uff1a\n]{0,80}?[:：]"
)
_SPEECH_ACTION_RE = re.compile(
    r"(?:says?|said|speaks?|spoke|speaking|utters?|uttered|"
    r"whispers?|whispered|murmurs?|murmured|replies?|replied|"
    r"asks?|asked|answers?|answered|"
    r"\u8bf4|\u8bf4\u9053|\u5f00\u53e3|\u56de\u7b54|\u95ee|"
    r"\u558a|\u4f4e\u58f0\u8bf4|\u5599\u5583\u8bf4)",
    re.IGNORECASE,
)
_SPEAKER_CLAUSE_SPLIT_RE = re.compile(
    r"[\u201c\u201d\"\u3002\uff01\uff1f.!?;；\n]"
)


class _DialogueCue(NamedTuple):
    speaker: str
    text: str


def _normalize_dialogue_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _normalize_dialogue_speaker(value: str) -> str:
    speaker = _normalize_dialogue_text(value)
    speaker = re.sub(
        r"^\[(?:\u8bf4\u8bdd\u4eba|Speaker)\]",
        "",
        speaker,
        flags=re.IGNORECASE,
    )
    speaker = re.sub(r"[\uff08(][^\uff09)]*[\uff09)]$", "", speaker)
    return speaker


def _extract_audio_dialogue_cues(audio_desc: str) -> list[_DialogueCue]:
    """Extract dialogue text together with its explicit speaker label."""
    cues: list[_DialogueCue] = []
    for line in str(audio_desc or "").splitlines():
        if not line.strip():
            continue
        has_speaker_marker = "[说话人]" in line or "[Speaker]" in line
        has_markerless_speaker = bool(_MARKERLESS_SPEAKER_RE.search(line))
        if not has_speaker_marker and not has_markerless_speaker:
            continue
        for match in _AUDIO_DIALOGUE_RE.finditer(line):
            text = match.group("text").strip()
            if not text:
                continue
            cues.append(
                _DialogueCue(
                    speaker=_normalize_dialogue_speaker(match.group("speaker")),
                    text=text,
                )
            )
    return cues


def _extract_audio_dialogue(audio_desc: str) -> list[str]:
    """Extract spoken lines without treating ambient sound descriptions as dialogue."""
    return [cue.text for cue in _extract_audio_dialogue_cues(audio_desc)]


def _dialogue_speaker_aliases(cues: list[_DialogueCue]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for cue in cues:
        speaker = _normalize_dialogue_speaker(cue.speaker)
        if speaker:
            aliases.setdefault(speaker, set()).add(speaker)
    return aliases


def _explicit_speaker_for_occurrence(
    normalized_motion: str,
    position: int,
    text_length: int,
    aliases: dict[str, set[str]],
) -> str:
    """Return a uniquely named speaker near one dialogue occurrence, if any."""
    prefix = normalized_motion[max(0, position - 160):position]
    suffix = normalized_motion[
        position + text_length:position + text_length + 160
    ]
    prefix_clause = _SPEAKER_CLAUSE_SPLIT_RE.split(
        prefix.strip("\u201c\u201d\"")
    )[-1].casefold()
    suffix_clause = _SPEAKER_CLAUSE_SPLIT_RE.split(
        suffix.strip("\u201c\u201d\"")
    )[0].casefold()

    mentioned: set[str] = set()
    for speaker, speaker_aliases in aliases.items():
        for alias in speaker_aliases:
            normalized_alias = alias.casefold()
            for match in re.finditer(re.escape(normalized_alias), prefix_clause):
                if _SPEECH_ACTION_RE.search(prefix_clause[match.end():]):
                    mentioned.add(speaker)
            for match in re.finditer(re.escape(normalized_alias), suffix_clause):
                if _SPEECH_ACTION_RE.search(suffix_clause[:match.start()]):
                    mentioned.add(speaker)

    if len(mentioned) == 1:
        return next(iter(mentioned))
    return ""


def validate_dialogue_performance(audio_desc: str, motion_desc: str) -> None:
    """Require motion_desc to preserve audio_desc dialogue verbatim and in order."""
    cues = _extract_audio_dialogue_cues(audio_desc)
    if not cues:
        return

    expected = [cue.text for cue in cues]
    aliases = _dialogue_speaker_aliases(cues)
    normalized_motion = _normalize_dialogue_text(motion_desc)
    cursor = 0
    missing: list[str] = []
    for cue in cues:
        line = cue.text
        normalized_line = _normalize_dialogue_text(line)
        position = normalized_motion.find(normalized_line, cursor)
        if position < 0:
            if normalized_motion.find(normalized_line) >= 0:
                raise ValueError(
                    "motion_desc dialogue order or multiplicity differs from "
                    "audio_desc; "
                    f"expected order: {expected}"
                )
            missing.append(line)
            continue

        explicit_speaker = _explicit_speaker_for_occurrence(
            normalized_motion,
            position,
            len(normalized_line),
            aliases,
        )
        expected_speaker = _normalize_dialogue_speaker(cue.speaker)
        if (
            explicit_speaker
            and expected_speaker
            and explicit_speaker.casefold() != expected_speaker.casefold()
        ):
            raise ValueError(
                "motion_desc attributes dialogue to the wrong speaker; "
                f"expected {cue.speaker!r} for {line!r}, "
                f"found {explicit_speaker!r}"
            )
        cursor = position + len(normalized_line)

    if missing:
        raise ValueError(
            "motion_desc omitted dialogue from audio_desc; "
            f"missing lines: {missing}"
        )


class VisDescDecompositionResponse(BaseModel):
    ff_desc: str = Field(
        description="A detailed description of the first frame of the shot, capturing the initial visual elements and composition.",
        # examples=[
        #     "Medium shot of a supermarket aisle at eye level. Bob(a tall man wearing a blue shirt and jeans) is positioned on the right side of the frame, captured in profile and facing right, while Alice(a young woman with short hair, wearing a green dress) is on the left, shown pushing a shopping cart with her gaze lowered toward the ground. They are arranged in a front-to-back spatial relationship. Shelves line both sides of the frame, and cool-toned fluorescent lighting from above washes over the scene. The vibrant colors of product packaging contrast with the metallic gray of the shopping cart, all contained within a stable, horizontally balanced composition.",
        #     "Extreme long shot. Aerial view from hundreds of meters above the ground. The boundless golden desert resembles undulating frozen waves, occupying the vast majority of the frame. At the very center of the image, a tiny, solitary explorer appears only as a faint dark speck, dragging a long, lonely trail of footprints behind him, stretching all the way to the edge of the frame.",
        #     "Medium shot at eye level angle. Designer A(with a beard, wearing a white suit) leans forward passionately, speaking emphatically. Product Manager B(with a beard, wearing a white T-shirt) sits with crossed arms, looking skeptical. Between them, Development Engineer C(brown hair, wearing a blue T-shirt) appears anxious, glancing between the two. Project Manager D(curly hair, wearing a red T-shirt) prepares to mediate, focusing on a whiteboard. Bright overhead lighting highlights their expressions, with a blurred whiteboard and glass wall in the background.",
        #     "A low-angle close-up shot captures the figure from below, framing him from the chest up. His face appears resolute and commanding, his eyes piercing as he speaks passionately. Flecks of saliva are visible, emphasizing his intensity. The overcast sky breaks with occasional light, casting him as a heroic, almost monumental figure against the gloom.",
        #     "An extremely close-up of an old, motionless pocket watch. Soft light highlights scratches on its brass case and the enamel dial with Roman numerals. The second hand remains fixed at 'VIII', casting a sharp shadow. A wrinkled finger gently touches the glass surface, evoking a tangible sense of stillness and time.",
        #     "An over-the-shoulder shot at eye level, positioned behind Character A(red hair, wearing a white T-shirt). The foreground, including A's shoulder and head, is softly blurred, directing focus onto Character B(with a beard, wearing a white T-shirt)'s face. B's subtle reactions—shifting from surprise to confusion, then to a glimmer of understanding—are clearly visible. The café background is gently blurred with warm lighting.",
        # ]
    )
    ff_vis_char_idxs: List[int] = Field(
        description="A list of indices of characters that are visible in the first frame of the shot, corresponding to the character list provided in the input.",
        examples=[[0], [1], [0, 1], []]
    )
    lf_desc: str = Field(
        description="A detailed description of the last frame of the shot, capturing the concluding visual elements and composition.",
    )
    lf_vis_char_idxs: List[int] = Field(
        description="A list of indices of characters that are visible in the last frame of the shot, corresponding to the character list provided in the input.",
        examples=[[0], [1], [0, 1], []]
    )
    motion_desc: str = Field(
        description="The motion description of the shot. Describe the dynamic visual changes within the shot (camera movement and the movement of elements within the frame)",
        examples=[
            "Static camera. Alice (short hair, wearing a green dress) is walking towards the camera.",
            "Dolly in from meidum shot to close-up. Bob (with a beard, wearing a white T-shirt) smiles to the camera.",
        ]
    )
    variation_type: Literal["large", "medium", "small"] = Field(
        description="Indicates the degree of change between the first frame and the last frame.",
    )
    variation_reason: str = Field(
        description="The reason for the variation type of the shot.",
        examples=[
            "This is a smooth transition shot from the sky to the ground. The content of the shot has changed significantly, so the variation type is large.",
            "Compared to the first frame, a new character appears in the last frame, and there are no significant changes in the composition. So the variation type is medium.",
            "Compared to the first frame, there are only minor changes in the composition. So the variation type is small.",
            "This shot only shows Alice speaking and the changes in her facial expressions, thus the variation type is small.",
        ],
    )



class StoryboardArtist:
    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = chat_model
        self.last_director_interpretation = ""


    @retry(stop=stop_after_attempt(3), after=after_func)
    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: Optional[str] = None,
        retry_timeout: int = 150,
    ) -> List[ShotBriefDescription]:

        class StoryboardResponse(BaseModel):
            storyboard: List[ShotBriefDescription] = Field(
                description="A complete storyboard of the scene, including the visual and audio description of each shot.",
            )

        script_str = script.strip()
        characters_str = "\n".join([f"Character {index}: {char}" for index, char in enumerate(characters)])
        user_requirement_str = user_requirement.strip() if user_requirement else ""

        interpretation_messages = [
            ('system', system_prompt_template_interpret_scene),
            (
                'human',
                human_prompt_template_interpret_scene.format(
                    script_str=script_str,
                    characters_str=characters_str,
                    user_requirement_str=user_requirement_str,
                ),
            ),
        ]
        interpretation_response = await asyncio.wait_for(
            self.chat_model.ainvoke(interpretation_messages),
            timeout=retry_timeout,
        )
        director_interpretation = _message_content_to_text(
            getattr(interpretation_response, "content", interpretation_response)
        )
        if not director_interpretation:
            raise ValueError("Scene interpretation model returned empty text.")
        self.last_director_interpretation = director_interpretation

        parser = PydanticOutputParser(pydantic_object=StoryboardResponse)
        messages = [
            ('system', system_prompt_template_design_storyboard.format(format_instructions=parser.get_format_instructions())),
            (
                'human',
                human_prompt_template_design_storyboard.format(
                    script_str=script_str,
                    characters_str=characters_str,
                    user_requirement_str=user_requirement_str,
                    director_interpretation=director_interpretation,
                ),
            ),
        ]
        chain = self.chat_model | parser
        response: StoryboardResponse = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        storyboard = response.storyboard

        return storyboard




    @retry(stop=stop_after_attempt(3), after=after_func)
    async def decompose_visual_description(
        self,
        shot_brief_desc: ShotBriefDescription,
        characters: List[CharacterInScene],
        retry_timeout: int = 150,
        scene_context: str = "",
        previous_shot_context: str = "",
        next_shot_context: str = "",
        duration_seconds: Optional[float] = None,
        duration_status: Optional[str] = None,
    ) -> ShotDescription:
        parser = PydanticOutputParser(pydantic_object=VisDescDecompositionResponse)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt_template_decompose_visual_description),
                ('human', human_prompt_template_decompose_visual_description),
            ]
        )
        chain = prompt_template | self.chat_model | parser

        prompt_inputs = _build_decompose_prompt_inputs(
            shot_brief_desc=shot_brief_desc,
            characters=characters,
            scene_context=scene_context,
            previous_shot_context=previous_shot_context,
            next_shot_context=next_shot_context,
            duration_seconds=duration_seconds,
            duration_status=duration_status,
        )

        decomposition: VisDescDecompositionResponse = await asyncio.wait_for(
            chain.ainvoke(
                input={
                    "format_instructions": parser.get_format_instructions(),
                    **prompt_inputs,
                },
            ),
            timeout=retry_timeout,
        )

        validate_char_idxs(decomposition.ff_vis_char_idxs, len(characters), "ff_vis_char_idxs")
        validate_char_idxs(decomposition.lf_vis_char_idxs, len(characters), "lf_vis_char_idxs")
        validate_dialogue_performance(
            shot_brief_desc.audio_desc,
            decomposition.motion_desc,
        )

        return ShotDescription(
            idx=shot_brief_desc.idx,
            is_last=shot_brief_desc.is_last,
            cam_idx=shot_brief_desc.cam_idx,
            visual_desc=shot_brief_desc.visual_desc,
            variation_type=decomposition.variation_type,
            variation_reason=decomposition.variation_reason,
            ff_desc=decomposition.ff_desc,
            ff_vis_char_idxs=decomposition.ff_vis_char_idxs,
            lf_desc=decomposition.lf_desc,
            lf_vis_char_idxs=decomposition.lf_vis_char_idxs,
            motion_desc=decomposition.motion_desc,
            audio_desc=shot_brief_desc.audio_desc,
            estimated_duration_seconds=duration_seconds,
        )


def validate_char_idxs(idxs, num_characters, field_name):
    """Reject LLM-emitted character indices outside [0, num_characters).

    Negative values would silently select the wrong character via Python
    indexing; out-of-range values would crash deep inside the render gather.
    Raising here lets the @retry on decompose_visual_description re-ask.
    """
    invalid = [idx for idx in idxs if idx < 0 or idx >= num_characters]
    if invalid:
        raise ValueError(
            f"{field_name} contains invalid character indices {invalid}; "
            f"valid range is 0..{num_characters - 1}"
        )
