from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlparse


DEFAULT_V2_ROOT = Path(
    os.environ.get("VIMAX_V2_ROOT", "aigc-production-platform/Agent")
)
DEFAULT_PLACEHOLDER_DURATION_SECONDS = 4.0
AIRBORNE_SUPPORT_PLACEHOLDERS = {
    "air",
    "airborne",
    "midair",
    "in the air",
    "in flight",
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Expected a model or mapping, got {type(value).__name__}")


def _load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_MODULE_CACHE: dict[Path, dict[str, Any]] = {}


def _load_v2_modules(v2_root: str | Path) -> dict[str, Any]:
    """Import the existing V2 implementation without modifying or copying it."""
    root = Path(v2_root).resolve()
    if root in _MODULE_CACHE:
        return _MODULE_CACHE[root]
    if not (root / "agent").is_dir():
        raise FileNotFoundError(f"V2 Agent package not found under: {root}")

    _load_env_file(root / ".env")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    modules = {
        "storyboard_ir": importlib.import_module("agent.schema.storyboard_ir"),
        "planner_h": importlib.import_module("agent.harness.planners.planner_h"),
        "production": importlib.import_module("agent.harness.production"),
        "openai_compat": importlib.import_module("agent.llm.openai_compat"),
    }
    required = (
        "AssetRef",
        "CameraSpec",
        "MotionBeat",
        "SceneIR",
        "ShotIR",
        "StoryboardIR",
    )
    for name in required:
        if not hasattr(modules["storyboard_ir"], name):
            raise RuntimeError(f"V2 schema is missing required symbol: {name}")
    for name in ("planner_h_action_director_draft", "build_action_director_prompt"):
        if not hasattr(modules["planner_h"], name):
            raise RuntimeError(f"V2 planner_h is missing required symbol: {name}")
    for name in ("choreography_advisories", "choreography_blockers"):
        if not hasattr(modules["production"], name):
            raise RuntimeError(f"V2 production is missing required symbol: {name}")
    _MODULE_CACHE[root] = modules
    return modules


def _normalize_duration_context(
    decomposition_contexts: list[dict[str, Any]] | dict[Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(decomposition_contexts, dict):
        items = decomposition_contexts.values()
    else:
        items = decomposition_contexts
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        shot_idx = item.get("shot_idx")
        if shot_idx is None:
            continue
        raw_duration = item.get("duration_seconds")
        duration: float | None
        try:
            duration = float(raw_duration) if raw_duration is not None else None
        except (TypeError, ValueError):
            duration = None
        if duration is not None and duration <= 0:
            duration = None
        normalized.append(
            {
                "shot_idx": int(shot_idx),
                "duration_seconds": duration,
                "duration_status": (
                    item.get("duration_status")
                    or ("provided" if duration is not None else "not_provided")
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["shot_idx"])


def _duration_source_label(status: str | None) -> str:
    if status == "provided":
        return "production duration provided"
    if status == "director_estimated":
        return "director-estimated duration; production timing not verified"
    if status == "partial":
        return "some durations provided; missing values are internal placeholders"
    if status == "not_provided":
        return "production duration not provided; internal placeholder"
    return "duration source unspecified"


def _build_canonicalization_metadata(
    original_blockers: list[str],
    structural_blockers: list[str],
) -> dict[str, Any]:
    structural_clear = not structural_blockers
    return {
        "original_structural_blockers": original_blockers,
        "structural_blockers": structural_blockers,
        "structural_status": "PASS" if structural_clear else "FAIL",
        "structural_blockers_clear": structural_clear,
        "text_continuity_status": "FAIL" if structural_blockers else "NOT VERIFIED",
        "production_ready": False,
        "production_ready_reason": (
            "Structural blocker clearance does not verify semantic continuity, "
            "spatial continuity, timing, dialogue-performance synchronization, "
            "or directorial quality. Human review is required."
        ),
    }


def build_v2_storyboard_ir(
    *,
    script: str,
    characters: list[dict[str, Any]],
    storyboard: list[dict[str, Any]],
    shot_descriptions: list[dict[str, Any]],
    decomposition_contexts: list[dict[str, Any]] | dict[Any, dict[str, Any]],
    v2_root: str | Path = DEFAULT_V2_ROOT,
):
    """Map ViMax director artifacts into V2's existing StoryboardIR contract."""
    modules = _load_v2_modules(v2_root)
    schema = modules["storyboard_ir"]
    AssetRef = schema.AssetRef
    CameraSpec = schema.CameraSpec
    MotionBeat = schema.MotionBeat
    SceneIR = schema.SceneIR
    ShotIR = schema.ShotIR
    StoryboardIR = schema.StoryboardIR

    ordered_characters = sorted(characters, key=lambda item: int(item["idx"]))
    asset_manifest = [
        AssetRef(
            kind="character",
            name=str(item.get("identifier_in_scene") or f"Character {item['idx']}"),
            look="; ".join(
                part
                for part in (
                    str(item.get("static_features") or "").strip(),
                    str(item.get("dynamic_features") or "").strip(),
                )
                if part
            ),
            role="principal",
        )
        for item in ordered_characters
    ]
    asset_by_character_idx = {
        int(item["idx"]): asset
        for item, asset in zip(ordered_characters, asset_manifest)
    }

    contexts = _normalize_duration_context(decomposition_contexts)
    duration_by_shot_idx = {
        int(item["shot_idx"]): item["duration_seconds"] for item in contexts
    }
    provided = [
        duration
        for duration in duration_by_shot_idx.values()
        if duration is not None
    ]
    context_statuses = [item.get("duration_status") for item in contexts]
    if (
        contexts
        and len(contexts) == len(storyboard)
        and all(
            item.get("duration_status") == "provided"
            and item.get("duration_seconds") is not None
            for item in contexts
        )
    ):
        duration_source = "provided"
    elif (
        any(
            item.get("duration_status") == "director_estimated"
            and item.get("duration_seconds") is not None
            for item in contexts
        )
        and not any(status == "provided" for status in context_statuses)
    ):
        duration_source = "director_estimated"
    elif provided:
        duration_source = "partial"
    else:
        duration_source = "not_provided"

    shot_desc_by_idx = {
        int(item["idx"]): item for item in shot_descriptions
    }
    ordered_storyboard = sorted(storyboard, key=lambda item: int(item["idx"]))
    shots = []
    for item in ordered_storyboard:
        idx = int(item["idx"])
        detail = shot_desc_by_idx.get(idx)
        if detail is None:
            raise ValueError(f"Missing shot_description for shot {idx}")
        duration = duration_by_shot_idx.get(idx)
        duration_seconds = (
            float(duration)
            if duration is not None and float(duration) > 0
            else DEFAULT_PLACEHOLDER_DURATION_SECONDS
        )
        visible_character_idxs = sorted(
            {
                int(value)
                for field in ("ff_vis_char_idxs", "lf_vis_char_idxs")
                for value in (detail.get(field) or [])
            }
        )
        asset_refs = [
            asset_by_character_idx[value]
            for value in visible_character_idxs
            if value in asset_by_character_idx
        ]
        order = idx + 1
        shots.append(
            ShotIR(
                shot_id=f"seg01-sh{order:02d}",
                order=order,
                duration_seconds=duration_seconds,
                purpose=str(item.get("visual_desc") or ""),
                cut_motivation=str(
                    detail.get("variation_reason")
                    or item.get("variation_type")
                    or ""
                ),
                entry_state=str(detail.get("ff_desc") or ""),
                exit_state=str(detail.get("lf_desc") or ""),
                camera=CameraSpec(),
                motion_beats=[
                    MotionBeat(
                        at_seconds=0.0,
                        duration_seconds=duration_seconds,
                        action=str(detail.get("motion_desc") or ""),
                    )
                ],
                asset_refs=asset_refs,
            )
        )

    scene_id = "seg01-sc01"
    scene = SceneIR(
        scene_id=scene_id,
        order=1,
        label="ViMax run_final_v3 director plan",
        start_seconds=0.0,
        end_seconds=sum(shot.duration_seconds for shot in shots),
        shots=shots,
    )
    return StoryboardIR(
        segment_index=1,
        project_id="vimax-v2-action-bridge",
        title="ViMax director plan action bridge",
        source_script_version=hashlib.sha256(script.encode("utf-8")).hexdigest()[:16],
        scenes=[scene],
        asset_manifest=asset_manifest,
        notes={
            "duration_source": duration_source,
            "duration_validation": "not_verified",
            "duration_note": _duration_note(duration_source),
            "duration_context": contexts,
            "source": "ViMax run_final_v3 text director artifacts",
        },
    )


def _duration_note(duration_source: str) -> str:
    if duration_source == "provided":
        return (
            "ViMax supplied user-provided hard durations, but performance timing "
            "has not been independently validated."
        )
    if duration_source == "director_estimated":
        return (
            "ViMax supplied director estimate durations for internal planning; "
            "each is a director estimate, not a user-specified hard duration, and "
            "production timing is not verified."
        )
    if duration_source == "partial":
        return (
            "ViMax supplied a mix of explicit and estimated durations. Missing or "
            "estimated values are not verified production timing."
        )
    return (
        "V2 requires positive durations internally. Missing ViMax durations use "
        "a placeholder and must not be treated as validated timing."
    )


def augment_action_prompt(
    base_prompt: str,
    *,
    script: str,
    duration_context: list[dict[str, Any]],
) -> str:
    """Append authoritative source material once, without replacing V2's prompt."""
    duration_lines = []
    for item in _normalize_duration_context(duration_context):
        duration = item["duration_seconds"]
        status = item.get("duration_status")
        if duration is None:
            duration_lines.append(
                f"- Shot {item['shot_idx']}: duration not provided."
            )
        elif status == "director_estimated":
            duration_lines.append(
                f"- Shot {item['shot_idx']}: {duration:g} seconds is a director "
                "estimate, not a user-specified hard duration."
            )
        elif status == "provided":
            duration_lines.append(
                f"- Shot {item['shot_idx']}: {duration:g} seconds is a "
                "user-provided hard duration."
            )
        else:
            duration_lines.append(
                f"- Shot {item['shot_idx']}: {duration:g} seconds provided with "
                "unspecified source."
            )
    if not duration_lines:
        duration_lines.append("- No shot durations were provided.")
    return (
        base_prompt.rstrip()
        + "\n\n<SOURCE_SCRIPT>\n"
        + script.strip()
        + "\n</SOURCE_SCRIPT>\n\n"
        + "<SHOT_DURATION_CONTEXT>\n"
        + "\n".join(duration_lines)
        + "\n</SHOT_DURATION_CONTEXT>\n\n"
        + "The source script above is authoritative. Preserve its events, dialogue, "
        "character goals, and existing props; do not add a new event or prop. "
        "If the script includes a scripted acquisition or retrieval, do not remove "
        "that action; represent the acquisition or account for it with a clear "
        "off-screen action or time ellipsis. Do not infer occupation-typical "
        "accessories or tools unless they are present in the "
        "script or supplied character definitions. "
        "Where a duration is not provided, do not invent one and do not claim "
        "timing validation."
    )


def canonicalize_choreography_draft(draft: Any) -> Any:
    """Map explicit airborne support placeholders to V2's not-held contract.

    V2's ``ObjectStateFrame.holder`` is derived from ``support``. A literal
    value such as ``air`` is not a support surface or a hand, so V2 currently
    treats it as a phantom holder named "air". Only exact, structured
    placeholder values are normalized here; prose is never inspected.
    """
    canonical = draft.model_copy(deep=True)
    shots = []
    for shot in canonical.shots:
        events = []
        for event in shot.macro_events:
            steps = []
            for step in event.steps:
                state = step.object_state_after
                if (
                    state is not None
                    and (state.support or "").strip().casefold()
                    in AIRBORNE_SUPPORT_PLACEHOLDERS
                ):
                    step = step.model_copy(
                        update={
                            "object_state_after": state.model_copy(
                                update={"support": ""}
                            )
                        }
                    )
                steps.append(step)
            events.append(event.model_copy(update={"steps": steps}))
        shots.append(shot.model_copy(update={"macro_events": events}))
    return canonical.model_copy(update={"shots": shots})


class SourceScriptProvider:
    """Thin wrapper around the original V2 structured-output provider."""

    def __init__(
        self,
        *,
        delegate: Any,
        script: str,
        duration_context: list[dict[str, Any]],
    ) -> None:
        self.delegate = delegate
        self.script = script
        self.duration_context = list(duration_context)
        self.base_prompt = ""
        self.final_prompt = ""

    async def generate_structured(self, prompt: str, schema: Any, **kwargs: Any):
        self.base_prompt = prompt
        self.final_prompt = augment_action_prompt(
            prompt,
            script=self.script,
            duration_context=self.duration_context,
        )
        return await self.delegate.generate_structured(
            self.final_prompt,
            schema,
            **kwargs,
        )


def _create_capturing_provider(v2_root: str | Path) -> Any:
    modules = _load_v2_modules(v2_root)
    OpenAICompatProvider = modules["openai_compat"].OpenAICompatProvider

    class UsageCapturingOpenAICompatProvider(OpenAICompatProvider):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.calls: list[dict[str, Any]] = []
            self.raw_responses: list[str] = []

        async def _call_any(self, messages, schema_dump, temperature, schema_name: str):
            started = time.perf_counter()
            try:
                response = await super()._call_any(
                    messages,
                    schema_dump,
                    temperature,
                    schema_name,
                )
            except Exception as exc:
                self.calls.append(
                    {
                        "schema": schema_name,
                        "elapsed_seconds": round(time.perf_counter() - started, 4),
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                        "error": str(exc),
                    }
                )
                raise
            content = response.choices[0].message.content or ""
            usage = _usage_from_response(response)
            self.calls.append(
                {
                    "schema": schema_name,
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                    "usage": usage,
                    "model": getattr(response, "model", self._model),
                }
            )
            self.raw_responses.append(content)
            return response

    return UsageCapturingOpenAICompatProvider()


def _usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", 0)
        or 0
    )
    output_tokens = int(
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", 0)
        or 0
    )
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _strip_json_fences(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _summarize_usage(provider: Any, *, started_at: datetime, started: float,
                     model: str, base_url: str) -> dict[str, Any]:
    calls = list(getattr(provider, "calls", []))
    input_tokens = sum(call["usage"]["input_tokens"] for call in calls)
    output_tokens = sum(call["usage"]["output_tokens"] for call in calls)
    total_tokens = sum(call["usage"]["total_tokens"] for call in calls)
    return {
        "llm_calls": len(calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "calls": calls,
        "started_at_utc": started_at.isoformat(),
        "wall_time_seconds": round(time.perf_counter() - started, 4),
        "model": model,
        "provider_base_url_host": urlparse(base_url).netloc,
        "provider_billed_cost": None,
        "cost_note": (
            "The provider did not return a billed-cost amount; no cost is inferred."
        ),
        "image_generation_called": False,
        "video_generation_called": False,
    }


def _markdown_cell(value: Any) -> str:
    return (
        ("" if value is None else str(value))
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", "<br>")
        .strip()
    )


def write_action_report(
    *,
    output_path: str | Path,
    directed_ir: dict[str, Any],
    shot_descriptions: list[dict[str, Any]],
    blockers: list[str],
    advisories: list[str],
    usage: dict[str, Any],
    duration_context: list[dict[str, Any]] | dict[Any, dict[str, Any]] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_by_idx = {
        int(item["idx"]): item for item in shot_descriptions
    }
    duration_status_by_idx = {
        item["shot_idx"]: item["duration_status"]
        for item in _normalize_duration_context(duration_context or [])
    }
    duration_statuses = set(duration_status_by_idx.values())
    if duration_statuses == {"director_estimated"}:
        timing_boundary = (
            "- Timing: NOT VERIFIED. ViMax supplied director-estimated durations; "
            "these are planning estimates, not internally substituted `4.0s` "
            "placeholders. Production timing remains unverified."
        )
    elif "not_provided" in duration_statuses:
        timing_boundary = (
            "- Timing: NOT VERIFIED. Missing ViMax durations use V2's internal "
            "positive-duration placeholder and must not be treated as production "
            "timing."
        )
    elif duration_statuses == {"provided"}:
        timing_boundary = (
            "- Timing: NOT VERIFIED. ViMax supplied production durations, but "
            "performance timing has not been independently validated."
        )
    else:
        timing_boundary = (
            "- Timing: NOT VERIFIED. Duration coverage is incomplete or mixed; "
            "do not treat internal values as production timing."
        )
    shots = [
        shot
        for scene in directed_ir.get("scenes", [])
        for shot in scene.get("shots", [])
    ]
    structural_status = "PASS" if not blockers else "FAIL"
    text_continuity_status = "FAIL" if blockers else "NOT VERIFIED"
    lines = [
        "# ViMax to V2 Action Continuity Report",
        "",
        "Structural validation only. This report does not turn structural checks "
        "into a semantic or cinematographic PASS.",
        f"STRUCTURAL: {structural_status}",
        f"TEXT CONTINUITY: {text_continuity_status}",
        "PRODUCTION READY: NO",
        "TIMING: NOT VERIFIED",
        (
            "Production delivery status: BLOCKED. Resolve the structural blockers "
            "before human continuity review."
            if blockers
            else "Production delivery status: structural validation passed, but "
            "semantic and cinematographic readiness remains unverified."
        ),
        "",
        "## Shot Table",
        "",
        "| Shot | V2 shot ID | First frame | Motion | Last frame |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for shot in shots:
        order = int(shot.get("order", 0))
        source = source_by_idx.get(order - 1, {})
        lines.append(
            "| Shot {idx} | {shot_id} | {first} | {motion} | {last} |".format(
                idx=order - 1,
                shot_id=_markdown_cell(shot.get("shot_id")),
                first=_markdown_cell(
                    source.get("ff_desc") or shot.get("entry_state")
                ),
                motion=_markdown_cell(source.get("motion_desc")),
                last=_markdown_cell(
                    source.get("lf_desc") or shot.get("exit_state")
                ),
            )
        )

    lines.extend(["", "## Choreography And Object State Chain", ""])
    for shot in shots:
        lines.extend(
            [
                f"### {shot.get('shot_id', '')}",
                "",
                (
                    f"- Duration source value: `{shot.get('duration_seconds', '')}` "
                    f"({_duration_source_label(duration_status_by_idx.get(int(shot.get('order', 0)) - 1))})"
                ),
            ]
        )
        beats = shot.get("motion_beats") or []
        if not beats:
            lines.append("- Motion beats: none")
        for beat in beats:
            lines.append(
                f"- Macro event: {_markdown_cell(beat.get('action'))}"
            )
            for step in beat.get("steps") or []:
                state = step.get("object_state_after") or {}
                lines.append(
                    "  - Step {index}: actor={actor}; action={action}; "
                    "object={object}; interaction={interaction}; "
                    "state_after={state}".format(
                        index=step.get("index", ""),
                        actor=_markdown_cell(step.get("actor")),
                        action=_markdown_cell(step.get("action")),
                        object=_markdown_cell(step.get("object")),
                        interaction=_markdown_cell(step.get("interaction")),
                        state=_markdown_cell(
                            json.dumps(state, ensure_ascii=False)
                        ),
                    )
                )
        records = shot.get("object_states") or []
        if records:
            for record in records:
                lines.append(
                    "- Object state: {object_id}: before={before} -> after={after}".format(
                        object_id=record.get("object_id", ""),
                        before=_markdown_cell(
                            json.dumps(record.get("before") or {}, ensure_ascii=False)
                        ),
                        after=_markdown_cell(
                            json.dumps(record.get("after") or {}, ensure_ascii=False)
                        ),
                    )
                )
        else:
            lines.append("- Object state: no record emitted for this shot")
        lines.append("")

    lines.extend(["", "## Validator Output", "", "### BLOCK", ""])
    lines.extend(f"- {item}" for item in blockers)
    if not blockers:
        lines.append("- none")
    lines.extend(["", "### Advisory", ""])
    lines.extend(f"- {item}" for item in advisories)
    if not advisories:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Verification Boundary",
            "",
            f"- STRUCTURAL: {structural_status}. This only covers Pydantic, "
            "numbering, and validator execution.",
            f"- TEXT CONTINUITY: {text_continuity_status}. "
            "Semantic and cinematographic continuity cannot be promoted to PASS "
            "by Pydantic, numbering, or object-state validation alone.",
            "- Semantic continuity is not declared PASS; human review is required.",
            "- Prop continuity is limited to V2 object-state records and validator "
            "findings.",
            "- Spatial continuity: HUMAN REVIEW REQUIRED.",
            "- Dialogue/performance synchronization: HUMAN REVIEW REQUIRED.",
            timing_boundary,
            "- Director aesthetics: HUMAN REVIEW REQUIRED.",
            "- PRODUCTION READY: NO.",
            "",
            "## Usage",
            "",
            f"- LLM calls: {usage.get('llm_calls', 0)}",
            f"- Input tokens: {usage.get('input_tokens', 0)}",
            f"- Output tokens: {usage.get('output_tokens', 0)}",
            f"- Total tokens: {usage.get('total_tokens', 0)}",
            f"- Wall time seconds: {usage.get('wall_time_seconds', 0)}",
            f"- Billed cost: {usage.get('provider_billed_cost')}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _collect_object_state_records(directed_ir: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in directed_ir.get("scenes", []):
        for shot in scene.get("shots", []):
            for record in shot.get("object_states") or []:
                records.append(
                    {
                        "shot_id": shot.get("shot_id"),
                        "shot_order": shot.get("order"),
                        **record,
                    }
                )
    return records


async def run_bridge(
    *,
    source_run: str | Path,
    output_dir: str | Path,
    v2_root: str | Path = DEFAULT_V2_ROOT,
) -> dict[str, Any]:
    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    script = (source / "script.txt").read_text(encoding="utf-8")
    characters = json.loads(
        (source / "characters.json").read_text(encoding="utf-8")
    )
    storyboard = json.loads(
        (source / "storyboard.json").read_text(encoding="utf-8")
    )
    contexts = json.loads(
        (source / "decomposition_contexts.json").read_text(encoding="utf-8")
    )
    shot_descriptions = [
        json.loads(
            (source / "shots" / str(item["idx"]) / "shot_description.json").read_text(
                encoding="utf-8"
            )
        )
        for item in sorted(storyboard, key=lambda value: int(value["idx"]))
    ]

    output.mkdir(parents=True, exist_ok=True)
    ir = build_v2_storyboard_ir(
        script=script,
        characters=characters,
        storyboard=storyboard,
        shot_descriptions=shot_descriptions,
        decomposition_contexts=contexts,
        v2_root=v2_root,
    )
    _write_json(output / "v2_storyboard_ir_input.json", ir.model_dump(mode="json"))

    modules = _load_v2_modules(v2_root)
    provider = _create_capturing_provider(v2_root)
    wrapped = SourceScriptProvider(
        delegate=provider,
        script=script,
        duration_context=_normalize_duration_context(contexts),
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    original_directed, draft = await modules["planner_h"].planner_h_action_director_draft(
        wrapped,
        ir,
    )
    elapsed_seconds = time.perf_counter() - started

    canonical_draft = canonicalize_choreography_draft(draft)
    directed = modules["planner_h"].apply_action_director(ir, canonical_draft)
    original_directed_payload = original_directed.model_dump(mode="json")
    directed_payload = directed.model_dump(mode="json")
    draft_payload = draft.model_dump(mode="json")
    canonical_draft_payload = canonical_draft.model_dump(mode="json")
    original_blockers = modules["production"].choreography_blockers(
        original_directed, 1
    )
    original_advisories = modules["production"].choreography_advisories(
        original_directed, 1
    )
    blockers = modules["production"].choreography_blockers(directed, 1)
    advisories = modules["production"].choreography_advisories(directed, 1)
    original_validation_raw = (
        original_directed_payload.get("notes", {}).get("choreography", {})
    )
    validation_raw = directed_payload.get("notes", {}).get("choreography", {})
    object_state_records = _collect_object_state_records(directed_payload)

    base_url = os.environ.get("AGENT_LLM_BASE_URL", "")
    model = os.environ.get("AGENT_LLM_MODEL", "")
    usage = _summarize_usage(
        provider,
        started_at=started_at,
        started=started,
        model=model,
        base_url=base_url,
    )
    usage["wall_time_seconds"] = round(elapsed_seconds, 4)

    (output / "v2_action_director_prompt.txt").write_text(
        wrapped.final_prompt,
        encoding="utf-8",
    )
    raw_content = provider.raw_responses[-1] if provider.raw_responses else ""
    try:
        raw_json = json.loads(_strip_json_fences(raw_content))
    except json.JSONDecodeError:
        raw_json = {"raw_content": raw_content, "parse_error": True}
    _write_json(output / "v2_choreography_raw_response.json", raw_json)
    _write_json(output / "v2_choreography_draft.json", draft_payload)
    _write_json(
        output / "v2_choreography_draft_canonicalized.json",
        canonical_draft_payload,
    )
    _write_json(
        output / "v2_directed_storyboard_ir_original.json",
        original_directed_payload,
    )
    _write_json(output / "v2_directed_storyboard_ir.json", directed_payload)
    _write_json(output / "v2_object_state_records.json", object_state_records)
    _write_json(
        output / "v2_validation_raw_original.json",
        original_validation_raw,
    )
    _write_json(output / "v2_validation_raw.json", validation_raw)
    _write_json(output / "v2_blockers_original.json", original_blockers)
    _write_json(output / "v2_blockers.json", blockers)
    _write_json(output / "v2_advisories_original.json", original_advisories)
    _write_json(output / "v2_advisories.json", advisories)
    _write_json(
        output / "v2_canonicalization.json",
        {
            "airborne_support_placeholders": sorted(
                AIRBORNE_SUPPORT_PLACEHOLDERS
            ),
            "source": "ViMax-to-V2 bridge field mapping",
            "original_draft": draft_payload,
            "canonical_draft": canonical_draft_payload,
            **_build_canonicalization_metadata(original_blockers, blockers),
        },
    )
    _write_json(output / "llm_usage.json", usage)
    write_action_report(
        output_path=output / "action_continuity_report.md",
        directed_ir=directed_payload,
        shot_descriptions=shot_descriptions,
        blockers=blockers,
        advisories=advisories,
        usage=usage,
        duration_context=contexts,
    )
    return {
        "output_dir": str(output),
        "blockers": blockers,
        "advisories": advisories,
        "usage": usage,
        "directed_ir": directed_payload,
        "draft": draft_payload,
        "shot_descriptions": shot_descriptions,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map ViMax director text artifacts into the existing V2 Action "
            "Director and Object Lifecycle validator."
        )
    )
    parser.add_argument(
        "--source-run",
        default=os.environ.get(
            "VIMAX_SOURCE_RUN",
            "deliverables/vimax_director_text_plan/run_final_v3",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "VIMAX_BRIDGE_OUTPUT_DIR",
            "deliverables/vimax_director_text_plan/run_final_v3_action_bridge",
        ),
    )
    parser.add_argument("--v2-root", default=str(DEFAULT_V2_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = asyncio.run(
        run_bridge(
            source_run=args.source_run,
            output_dir=args.output_dir,
            v2_root=args.v2_root,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "llm_calls": result["usage"]["llm_calls"],
                "blockers": len(result["blockers"]),
                "advisories": len(result["advisories"]),
                "wall_time_seconds": result["usage"]["wall_time_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
