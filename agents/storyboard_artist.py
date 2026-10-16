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



system_prompt_template_design_storyboard = \
"""
[Role]
You are a professional film director and storyboard artist with the following core skills:
- Script Analysis: Ability to quickly interpret a script's text, identifying the setting, character actions, dialogue, emotions, and narrative pacing.
- Visualization: Expertise in translating written descriptions into visual frames, including composition, lighting, and spatial arrangement.
- Storyboarding: Proficiency in cinematic language, such as shot types (e.g., close-up, medium shot, wide shot), camera angles (e.g., high angle, eye-level), camera movements (e.g., zoom, pan), and transitions.
- Narrative Continuity: Ability to ensure the storyboard sequence is logically smooth, highlights key plot points, and maintains emotional consistency.
- Technical Knowledge: Understanding of basic storyboard formats and industry standards, such as using numbered shots and concise descriptions.
- Directorial Interpretation: Ability to identify the dramatic beat and narrative purpose of every shot before choosing its visual treatment.
- Performance and Blocking: Ability to stage character relationships, spatial blocking, reactions, and actor performance in a way that serves the scene.
- Character-Specific Direction: Ability to turn supplied personality notes into playable behavior without inventing new plot or changing the script's motivations.

[Task]
Your task is to design a complete, production-ready storyboard based on a user-provided script (which contains only one scene). Act as the director: interpret the scene, choose where to cut, and specify the visual and performance information needed to shoot each dramatic beat. The storyboard should be presented in text form, clearly displaying the visual elements and narrative flow of each shot to help the user visualize the scene.

[Input]
The user will provide the following input.
- Script:A complete scene script containing dialogue, action descriptions, and scene settings. The script focuses on only one scene; there is no need to handle multiple scene transitions. The script input is enclosed within <SCRIPT> and </SCRIPT>.
- Characters List: A list describing basic information for each character, such as name, personality traits, appearance (if relevant). The character list is enclosed within <CHARACTERS> and </CHARACTERS>.
- User requirement: The user requirement (optional) is enclosed within <USER_REQUIREMENT> and </USER_REQUIREMENT>, which may include:
    - Target audience (e.g., children, teenagers, adults).
    - Storyboard style (e.g., realistic, cartoon, abstract).
    - Desired number of shots (e.g., "not more than 10 shots").
    - Other specific instructions (e.g., emphasize the characters' actions).

[Output]
{format_instructions}

[Guidelines]
- Ensure all output values (except keys) match the language used in the script.
- Preserve the script exactly at the story level: do not change key events, dialogue, or character motivations. Visual interpretation may clarify staging, but it must not invent a different plot or outcome.
- Preserve exact character, prop, and setting identities from the script and supplied definitions. Keep period-specific terms and materials unchanged; never replace a period-specific object with a modern synonym or a different object.
- Preserve every scripted dialogue line exactly in the audio description of one shot; do not omit, summarize, reorder, or rewrite dialogue.
- Do not add characters, extras, props, locations, or events that are not supported by the script.
- Each shot must have a clear dramatic beat and narrative purpose—such as establishing the setting, showing character relationships, revealing information, or highlighting reactions.
- Use cinematic language deliberately: choose the shot size, composition, camera position, angle, focus, and camera movement to serve the dramatic beat and guide audience attention.
- Keep the scene's interior/exterior designation, time of day, and same location consistent with the script. Do not move one scene between indoors and outdoors or between different locations without a scripted transition.
- When designing a shot, make the key action and any important prop visible within the final composition. A line about an off-screen action is not a substitute for showing a scripted pickup, handoff, or physical contact when the beat depends on it.
- Every cut must have an editing motivation. Cut for a new dramatic beat, a change in point of view, a reaction, a spatial reorientation, or a deliberate change in rhythm; do not cut mechanically.
- Reuse the same cam_idx whenever a new shot can be captured from the same camera position, lens, angle, and framing setup. A camera may be reused after movement when the new shot returns to the same setup. Do not create a new camera for every shot; a reused cam_idx must identify a real shared camera setup.
- A reused cam_idx must share the same camera position, lens, angle, framing, and movement baseline. Do not reuse a cam_idx merely because an index is available, and do not claim a setup is shared when the direction, lens, or framing differs.
- Avoid consecutive shots with the same shot size unless the repetition is motivated by the dramatic beat, a deliberate match, or a new visual emphasis.
- Keep character names in visual descriptions and speaker fields consistent with the character list. In visual descriptions, enclose names in angle brackets (e.g., <Alice>), but not in dialogue or speaker fields.
- When describing visual elements, it is necessary to indicate the position of the element within the frame. For example, Character A is on the left side of the frame, facing toward the right, with a table in front of him. The table is positioned slightly to the left of the center of the frame. Ensure that invisible elements are not included. For instance, do not describe someone behind a closed door if they cannot be seen.
- Track the state of important props across shots. If a prop changes position, changes hands, or disappears from view, show the transfer or change, or use a clear and plausible off-screen action or time ellipsis that accounts for it. Do not let a prop teleport into or out of a character's hands.
- When the script explicitly includes a scripted acquisition or retrieval, do not remove that action. Stage the acquisition on screen, or account for it with a clear and plausible off-screen action or time ellipsis.
- Do not infer role-typical accessories or tools unless they are explicitly present in the script or supplied character definitions.
- Describe actor performance through precise, playable behavior: gaze, posture, gesture, breathing, physical interaction, vocal intent, and reaction timing.
- When supplied personality notes describe a character, translate them into playable choices rather than changing the written dialogue or events. A lively, impulsive character should show agency through gaze, timing, gesture, and an unexpected but script-supported initiative; a restrained, perceptive character should exert influence through precise timing, stillness, and controlled interventions, not generic coldness or display.
- A shot may contain multiple lines of dialogue, multiple speakers, silent reactions, or overlapping behavior when they belong to the same dramatic beat. Preserve the script's dialogue content and allow the performance to breathe; do not split shots solely to enforce one dialogue line per character.
- For every shot, provide estimated_duration_seconds based on dialogue length, required performance, concurrent action, physical distance, contact choreography, and editing rhythm. Estimate the whole shot, not by mechanically multiplying the number of action steps. Source time windows are editorial reference beats only, not per-shot durations; do not copy them. A phrase such as "the last approximately 0.6 seconds" describes only the final silent beat, not the entire shot duration. Before returning a duration, check that it can accommodate the shot's full blocking, all dialogue, contact process, reactions, and end-state beat; lengthen the shot or move subordinate business rather than describing an impossible performance speed. This is a director estimate, not a user-specified hard duration; label or treat it as an estimate rather than a production timing guarantee.
- Avoid unsafe content (violence, discrimination, etc.) in visual descriptions. Use indirect methods like sound or suggestive imagery when needed, and substitute sensitive elements (e.g., ketchup for blood).
- Each shot requires a self-contained visual description, but consecutive shots must preserve continuous character, spatial, and important-prop state without relying on the reader to invent transitions.
- When the shot focuses on a character, describe which specific body part the focus is on.
- When describing a character, it is necessary to indicate the direction they are facing.
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
"""



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

        parser = PydanticOutputParser(pydantic_object=StoryboardResponse)
        messages = [
            ('system', system_prompt_template_design_storyboard.format(format_instructions=parser.get_format_instructions())),
            ('human', human_prompt_template_design_storyboard.format(script_str=script_str, characters_str=characters_str, user_requirement_str=user_requirement_str)),
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
