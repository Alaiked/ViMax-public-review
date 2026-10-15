"""ViMax/V2 -> H3 production task compiler.

This module is intentionally an adapter.  It consumes the existing ViMax
director artifacts and V2 choreography/state artifacts, then calls the
production platform's own materialization, prompt, recipe, request, and job
contract code.  It never submits a job and never invents asset identifiers.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator


PLATFORM_FUNCTIONS = [
    "agent.harness.director.materialize.materialize",
    "agent.harness.production.build_production_prompt",
    "agent.h3_contract.h3_aspect_ratio",
    "agent.h3_contract.compute_video_size",
    "backend.app.h3_recipe.compile_recipe",
    "agent.prompt.request_builder.GenerationRequestBuilder.build",
    "backend.app.routers.jobs.CreateJobRequest.model_validate",
]

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_DIALOGUE_RE = re.compile(
    r"\[\u8bf4\u8bdd\u4eba\]\s*([^\uff08(:\uff1a]+?)\s*"
    r"(?:\uff08[^\uff09]*\uff09|\([^)]*\))?\s*[:\uff1a]\s*"
    r"[\u201c\"](.+?)[\u201d\"]"
)
_SHOT_MARK_RE = re.compile(r"\[Shot\s+(\d+)\]")
_GLOBAL_PROMPT_TAIL_RE = re.compile(r"\n(?:overall_soundscape|non_diegetic_music):")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _clean_name(value: str) -> str:
    return str(value or "").strip()


def _extract_asset_id(text: str) -> str:
    match = _UUID_RE.search(str(text or ""))
    return match.group(0) if match else ""


def _character_bindings(characters: list[dict[str, Any]]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for item in characters:
        name = _clean_name(item.get("identifier_in_scene"))
        if not name:
            continue
        asset_id = _extract_asset_id(
            " ".join(
                [
                    str(item.get("static_features") or ""),
                    str(item.get("dynamic_features") or ""),
                ]
            )
        )
        if asset_id:
            bindings[name] = asset_id
    return bindings


def _extract_dialogues(
    audio_desc: str,
    name_to_entity_id: dict[str, str],
) -> list[dict[str, str]]:
    dialogues: list[dict[str, str]] = []
    for match in _DIALOGUE_RE.finditer(str(audio_desc or "")):
        speaker = _clean_name(match.group(1))
        text = _clean_name(match.group(2))
        if not text:
            continue
        dialogues.append(
            {
                "speaker_id": name_to_entity_id.get(speaker, speaker),
                "text": text,
            }
        )
    return dialogues


def _object_state_snapshots(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, str]]], list[dict[str, Any]]]:
    """Return per-shot snapshots and a simple inherited sequence.

    The snapshots are copied from V2 ObjectStateRecord output.  No state is
    inferred from prose here; missing records inherit the previous known
    after-state only for display/continuity reporting.
    """
    by_shot: dict[str, dict[str, dict[str, str]]] = {}
    for record in records:
        shot_id = _clean_name(record.get("shot_id"))
        object_id = _clean_name(record.get("object_id"))
        if not shot_id or not object_id:
            continue
        by_shot.setdefault(shot_id, {})[object_id] = {
            "before": dict(record.get("before") or {}),
            "after": dict(record.get("after") or {}),
        }
    return by_shot, records


def _camera_records(camera_tree: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for item in camera_tree:
        try:
            idx = int(item.get("idx"))
        except (TypeError, ValueError):
            continue
        records[idx] = item
    return records


def _camera_for_shot(
    cam_idx: int,
    camera_tree: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    item = camera_tree.get(cam_idx, {})
    parent = item.get("parent_cam_idx")
    parent_id = f"camera_{parent}" if parent is not None else None
    return {
        "cam_idx": cam_idx,
        "camera_id": f"camera_{cam_idx}",
        "parent_camera_id": parent_id,
        "active_shot_idxs": list(item.get("active_shot_idxs") or []),
        "reason": item.get("reason") or "",
        "parent_shot_idx": item.get("parent_shot_idx"),
        "is_parent_fully_covers_child": item.get("is_parent_fully_covers_child"),
        "missing_info": item.get("missing_info"),
    }


def _split_prompt_by_shot(prompt: str, shot_count: int) -> tuple[str, list[str]]:
    """Split the platform-rendered segment prompt into per-shot prompts.

    The platform renders one segment with `[Shot N]` markers.  H3 jobs are
    per-shot, so the common header is kept and each shot body is selected
    without re-rendering or rewriting the platform prompt.
    """
    matches = list(_SHOT_MARK_RE.finditer(prompt or ""))
    if len(matches) != shot_count:
        raise ValueError(
            "H3 segment prompt shot markers do not match the storyboard: "
            f"expected {shot_count}, found {len(matches)}"
        )
    preamble = prompt[: matches[0].start()].strip()
    sections: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        sections.append(prompt[match.start():end].strip())
    if sections:
        tail = _GLOBAL_PROMPT_TAIL_RE.search(sections[-1])
        if tail:
            sections[-1] = sections[-1][: tail.start()].strip()
    return preamble, sections


def _compose_shot_prompt(preamble: str, section: str) -> str:
    return f"{preamble}\n\n{section}".strip() if preamble else section.strip()


def _append_missing_dialogues(
    prompt: str,
    dialogues: list[dict[str, str]],
) -> str:
    """Restore dialogue lines dropped by the platform's one-line shot renderer."""
    missing: list[str] = []
    for line in dialogues:
        text = _clean_name(line.get("text"))
        if not text or text in prompt:
            continue
        speaker = _clean_name(line.get("speaker_id"))
        match = re.fullmatch(r"char_(\d+)", speaker)
        subject = f"<Subject {int(match.group(1))}>" if match else speaker
        missing.append(
            f"{subject} says on camera with synchronized lip movement, "
            f"<d>[Chinese]{text}</d>"
        )
    if not missing:
        return prompt
    return f"{prompt} {' '.join(missing)}".strip()


def _append_shot_soundscape(prompt: str, audio_desc: str) -> str:
    """Append only this shot's ambient lines; dialogue stays in `<d>` tags."""
    ambient_lines = [
        line.strip()
        for line in str(audio_desc or "").splitlines()
        if line.strip() and not line.strip().startswith("[\u8bf4\u8bdd\u4eba]")
    ]
    if not ambient_lines:
        return prompt
    return f"{prompt}\n\noverall_soundscape:\n" + "\n".join(ambient_lines)


def _make_director_plan(
    storyboard: list[dict[str, Any]],
    choreography: dict[str, Any],
    characters: list[dict[str, Any]],
    project_id: str,
):
    """Build the existing V2 DirectorPlan shape from real ViMax artifacts."""
    from agent.harness.director.schema import (
        CanonicalEntity,
        DirectorDialogue,
        DirectorPlan,
        DirectorShot,
        PerformancePlan,
        PerformanceShot,
        ScenePlan,
        SceneState,
        SceneStateFrame,
    )

    name_to_entity: dict[str, str] = {}
    entities: list[CanonicalEntity] = []
    for index, item in enumerate(characters, start=1):
        name = _clean_name(item.get("identifier_in_scene"))
        if not name:
            continue
        asset_id = _extract_asset_id(
            " ".join(
                [
                    str(item.get("static_features") or ""),
                    str(item.get("dynamic_features") or ""),
                ]
            )
        )
        entity_id = f"char_{index:02d}"
        name_to_entity[name] = entity_id
        entities.append(
            CanonicalEntity(
                entity_id=entity_id,
                kind="character",
                canonical_name=name,
                asset_id=asset_id,
                picture_slot=index,
            )
        )

    shots: list[DirectorShot] = []
    performance_shots: list[PerformanceShot] = []
    for index, item in enumerate(storyboard, start=1):
        shot_id = f"seg01-sh{index:02d}"
        choreo = next(
            (
                shot
                for shot in (choreography.get("shots") or [])
                if _clean_name(shot.get("shot_id")) == shot_id
            ),
            {},
        )
        dialogues = _extract_dialogues(
            str(item.get("audio_desc") or ""),
            name_to_entity,
        )
        shots.append(
            DirectorShot(
                shot_key=f"s01-k{index:02d}",
                order=index,
                duration_hint=float(item.get("estimated_duration_seconds") or 4.0),
                focus=str(item.get("visual_desc") or ""),
                camera={
                    "shot_size": str(choreo.get("shot_size") or ""),
                    "camera_movement": str(choreo.get("camera_trajectory") or ""),
                    "depth_of_field": str(choreo.get("depth_of_field") or ""),
                },
                entity_ids=list(name_to_entity.values()),
                actions=[str(item.get("visual_desc") or "")],
                dialogue=[
                    DirectorDialogue(speaker_id=line["speaker_id"], text=line["text"])
                    for line in dialogues
                ],
                state_before="T0",
                state_after="T1",
                supporting_behavior=str(choreo.get("supporting_behavior") or ""),
            )
        )
        performance_shots.append(
            PerformanceShot(
                shot_key=f"s01-k{index:02d}",
                macro_events=list(choreo.get("macro_events") or []),
                drop_actions=list(choreo.get("drop_actions") or []),
                supporting_behavior=str(choreo.get("supporting_behavior") or ""),
                camera_trajectory=str(choreo.get("camera_trajectory") or ""),
                shot_size=str(choreo.get("shot_size") or ""),
                depth_of_field=str(choreo.get("depth_of_field") or ""),
                notes=str(choreo.get("notes") or ""),
            )
        )

    # The state timeline is only a compatibility input to materialize.  The
    # authoritative cross-shot object records are retained separately below.
    object_states: dict[str, SceneStateFrame] = {}
    states = [
        SceneState(t="T0", objects={}),
        SceneState(t="T1", objects=object_states),
    ]
    plan = DirectorPlan(
        title="ViMax H3 production package",
        entities=entities,
        scenes=[
            ScenePlan(
                scene_id="seg01",
                label="ViMax director segment",
                summary="ViMax director plan compiled for H3",
                content_type="performance",
                shots=shots,
                states=states,
                soundscape=" ".join(
                    str(item.get("audio_desc") or "") for item in storyboard
                ),
                music="none",
            )
        ],
        script_meta={"project_id": project_id},
    )
    performance = PerformancePlan(
        scene_id="seg01",
        shots=performance_shots,
        notes="V2 choreography replayed through the existing materialize contract.",
    )
    return plan, performance


def _import_platform(platform_root: Path, runtime_root: Path):
    """Import only the existing platform functions required by this adapter."""
    root = platform_root.resolve()
    if not (root / "Agent" / "agent").is_dir():
        raise FileNotFoundError(f"platform Agent package not found: {root / 'Agent'}")
    agent_root = root / "Agent"
    added: list[str] = []
    for path in (str(root), str(agent_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    env_keys = (
        "AIGC_DB_PATH",
        "AIGC_UPLOADS_DIR",
        "AIGC_RESULTS_DIR",
        "AIGC_RELAY_DIR",
        "AIGC_THUMBS_DIR",
    )
    old_env = {key: os.environ.get(key) for key in env_keys}
    temp_root = runtime_root.resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    for key, child in (
        ("AIGC_DB_PATH", "aigc.sqlite3"),
        ("AIGC_UPLOADS_DIR", "uploads"),
        ("AIGC_RESULTS_DIR", "results"),
        ("AIGC_RELAY_DIR", "relay"),
        ("AIGC_THUMBS_DIR", "thumbs"),
    ):
        os.environ[key] = str(temp_root / child)
    try:
        from agent.harness.director import materialize
        from agent.harness.production import build_production_prompt
        from agent.h3_contract import compute_video_size, h3_aspect_ratio
        from agent.prompt.request_builder import GenerationRequestBuilder
        from agent.schema.generation import AssetRecord, AssetStatus, CapabilityMode
        from agent.schema.prompt_pipeline import CompiledPromptRecord, GenerationTarget
        from backend.app.h3_recipe import compile_recipe
        from backend.app.routers.jobs import CreateJobRequest

        return {
            "materialize": materialize.materialize,
            "build_production_prompt": build_production_prompt,
            "h3_aspect_ratio": h3_aspect_ratio,
            "compute_video_size": compute_video_size,
            "GenerationRequestBuilder": GenerationRequestBuilder,
            "AssetRecord": AssetRecord,
            "AssetStatus": AssetStatus,
            "CapabilityMode": CapabilityMode,
            "CompiledPromptRecord": CompiledPromptRecord,
            "GenerationTarget": GenerationTarget,
            "compile_recipe": compile_recipe,
            "CreateJobRequest": CreateJobRequest,
        }
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for path in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


def _build_task_assets(
    prompt_build: dict[str, Any],
    platform: dict[str, Any],
    project_id: str,
) -> list[Any]:
    records: list[Any] = []
    for index, ref in enumerate(prompt_build.get("asset_refs") or [], start=1):
        asset_id = _clean_name(ref.get("asset_id"))
        if not asset_id:
            continue
        name = _clean_name(ref.get("name")) or asset_id
        kind = _clean_name(ref.get("kind")) or "asset"
        records.append(
            platform["AssetRecord"](
                asset_id=asset_id,
                project_id=project_id,
                entity_ref=f"{kind}.{name}",
                kind="image",
                capability="",
                status=platform["AssetStatus"].APPROVED,
                metadata={
                    "display_name": name,
                    "asset_kind": kind,
                    "picture_slot": index,
                    "role": _clean_name(ref.get("role")),
                    "stage": _clean_name(ref.get("stage")),
                    "look": _clean_name(ref.get("look")),
                    "variant": _clean_name(ref.get("variant")),
                },
            )
        )

    manifest = (prompt_build.get("ir") or {}).get("asset_manifest") or []
    voice_names = {
        _clean_name(item.get("asset_id")): _clean_name(item.get("name"))
        for item in manifest
        if _clean_name(item.get("kind")) == "voice"
    }
    for index, raw_asset_id in enumerate(prompt_build.get("voice_ids") or [], start=1):
        asset_id = _clean_name(raw_asset_id)
        if not asset_id:
            continue
        name = voice_names.get(asset_id) or asset_id
        records.append(
            platform["AssetRecord"](
                asset_id=asset_id,
                project_id=project_id,
                entity_ref=f"voice.{name}",
                kind="audio",
                capability="",
                status=platform["AssetStatus"].APPROVED,
                metadata={
                    "display_name": name,
                    "asset_kind": "voice",
                    "voice_slot": index,
                },
            )
        )
    return records


def _asset_bindings_from_prompt_build(
    prompt_build: dict[str, Any],
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {
        "characters": {},
        "scene": {},
        "voice": {},
    }
    for ref in prompt_build.get("asset_refs") or []:
        asset_id = _clean_name(ref.get("asset_id"))
        if not asset_id:
            continue
        name = _clean_name(ref.get("name")) or asset_id
        kind = _clean_name(ref.get("kind"))
        if kind == "character":
            bindings["characters"][name] = asset_id
        elif kind == "scene":
            bindings["scene"][name] = asset_id

    manifest = (prompt_build.get("ir") or {}).get("asset_manifest") or []
    voice_names = {
        _clean_name(item.get("asset_id")): _clean_name(item.get("name"))
        for item in manifest
        if _clean_name(item.get("kind")) == "voice"
    }
    for raw_asset_id in prompt_build.get("voice_ids") or []:
        asset_id = _clean_name(raw_asset_id)
        if asset_id:
            bindings["voice"][voice_names.get(asset_id) or asset_id] = asset_id
    return bindings


def _request_payload(request: Any) -> dict[str, Any]:
    return request.model_dump(mode="json")


def compile_h3_production_package(
    *,
    director_run: str | Path,
    v2_run: str | Path,
    platform_root: str | Path,
    project_id: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compile the four-shot ViMax/V2 run into validated H3 task requests."""
    director_path = Path(director_run).resolve()
    v2_path = Path(v2_run).resolve()
    platform_path = Path(platform_root).resolve()
    if not director_path.is_dir():
        raise FileNotFoundError(f"director run not found: {director_path}")
    if not v2_path.is_dir():
        raise FileNotFoundError(f"V2 run not found: {v2_path}")

    storyboard = _read_json(director_path / "storyboard.json")
    camera_tree = _camera_records(_read_json(director_path / "camera_tree.json"))
    characters = _read_json(director_path / "characters.json")
    choreography = _read_json(v2_path / "v2_choreography_draft_canonicalized.json")
    object_records = _read_json(v2_path / "v2_object_state_records.json")

    runtime_root = (
        Path(output_dir).resolve() / "_platform_runtime"
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / ".manual_tmp" / "h3_import_probe"
    )
    platform = _import_platform(platform_path, runtime_root)
    asset_bindings = _character_bindings(characters)
    plan, performance = _make_director_plan(
        storyboard,
        choreography,
        characters,
        project_id,
    )
    parsed = platform["materialize"](plan, performance=[performance])
    prompt_build = platform["build_production_prompt"](
        parsed,
        project_id=project_id,
        segment_index=1,
    )
    segment_prompt = str(prompt_build.get("prompt") or "")
    preamble, shot_sections = _split_prompt_by_shot(segment_prompt, len(storyboard))
    object_by_shot, _ = _object_state_snapshots(object_records)
    task_assets = _build_task_assets(prompt_build, platform, project_id)
    platform_asset_bindings = _asset_bindings_from_prompt_build(prompt_build)
    platform_blockers = [
        str(issue) for issue in (prompt_build.get("blocking_issues") or []) if issue
    ]
    aspect_ratio = platform["h3_aspect_ratio"]()
    frame_width, frame_height = platform["compute_video_size"](aspect_ratio)

    tasks: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    prompt_compile_rows: list[dict[str, Any]] = []
    running_objects: dict[str, dict[str, dict[str, str]]] = {}
    for index, item in enumerate(storyboard):
        shot_id = f"seg01-sh{index + 1:02d}"
        director_duration = float(item.get("estimated_duration_seconds") or 0.0)
        h3_duration = max(1, int(math.ceil(director_duration)))
        cam_idx = int(item.get("cam_idx") or 0)
        camera = _camera_for_shot(cam_idx, camera_tree)
        dialogues = _extract_dialogues(
            str(item.get("audio_desc") or ""),
            {name: f"char_{i:02d}" for i, name in enumerate(asset_bindings, start=1)},
        )
        shot_prompt = _append_missing_dialogues(
            _compose_shot_prompt(preamble, shot_sections[index]),
            dialogues,
        )
        shot_prompt = _append_shot_soundscape(
            shot_prompt,
            str(item.get("audio_desc") or ""),
        )
        blocked_reasons = list(platform_blockers)
        if not prompt_build.get("ok") and not blocked_reasons:
            blocked_reasons.append("build_production_prompt_not_ok")

        # Copy V2 state records and carry the last known state across shots that
        # do not have a new ObjectStateRecord.  This is reporting only; it does
        # not mutate or replace the platform's choreography validation.
        current = object_by_shot.get(shot_id, {})
        for object_id, state in current.items():
            running_objects[object_id] = {
                "before": dict(state.get("before") or {}),
                "after": dict(state.get("after") or {}),
            }
        inherited = {
            object_id: {
                "before": dict(state.get("before") or {}),
                "after": dict(state.get("after") or {}),
            }
            for object_id, state in running_objects.items()
        }

        optimization = {
            "quality": "standard",
            "scene": "performance" if dialogues else "general",
            "finish": "default",
        }
        recipe = platform["compile_recipe"](
            "video.h3.generate",
            optimization,
            prompt_text=shot_prompt,
            mode="r2v",
            capability_group="multi_reference",
            ref_count=sum(1 for asset in task_assets if asset.kind == "image"),
        )
        target = platform["GenerationTarget"](
            target_id=f"h3-{shot_id}",
            entity_ref=f"shot.{shot_id}",
            target_type="shot",
            artifact_role="main_video",
            capability="video.h3.generate",
            mode=platform["CapabilityMode"].R2V,
            prompt_protocol="h3_full_reference_r2v",
        )
        record = platform["CompiledPromptRecord"](
            record_id=f"record-{shot_id}",
            generation_target_id=target.target_id,
            attempt_id=f"attempt-{shot_id}-1",
            entity_ref=target.entity_ref,
            capability="video.h3.generate",
            mode=platform["CapabilityMode"].R2V,
            prompt_protocol="h3_full_reference_r2v",
            output_contract="six_section_h3",
            structured_payload={"segment_prompt": segment_prompt, "shot_prompt": shot_prompt},
            rendered_prompt=shot_prompt,
            status="validated",
        )
        request = platform["GenerationRequestBuilder"]().build(
            record,
            target,
            attempt_number=1,
            assets=task_assets,
            project_id=project_id,
            params_extra={
                "duration": h3_duration,
                "aspect_ratio": aspect_ratio,
                "optimization": optimization,
                "capability_group": "multi_reference",
                "h3_backend": "auto",
            },
        )
        request_payload = _request_payload(request)
        validation_payload = {
            "capability": request_payload["capability"],
            "params": request_payload.get("params") or {},
            "inputs": request_payload.get("inputs") or {},
            "shot_id": request_payload.get("shot_id"),
            "source_client": request_payload.get("source_client") or "agent",
        }
        try:
            validated = platform["CreateJobRequest"].model_validate(validation_payload)
            valid = True
            errors: list[str] = []
        except Exception as exc:  # pragma: no cover - reported in the artifact
            validated = None
            valid = False
            errors = [str(exc)]
        if not valid:
            blocked_reasons.append("request_validation_failed")

        task = {
            "shot_id": shot_id,
            "shot_index": index + 1,
            "director_duration_seconds": director_duration,
            "h3_duration_seconds": h3_duration,
            "timing_status": "NOT VERIFIED",
            "camera": camera,
            "status": "BLOCKED" if blocked_reasons else "READY",
            "blocked_reasons": blocked_reasons,
            "prompt": shot_prompt,
            "dialogue": dialogues,
            "recipe": recipe,
            "request": request_payload,
            "inherited_object_states": inherited,
        }
        tasks.append(task)
        validation_rows.append(
            {
                "shot_id": shot_id,
                "valid": valid,
                "submitted": False,
                "errors": errors,
                "idempotency_key": request_payload.get("idempotency_key"),
                "request": (
                    validated.model_dump(mode="json") if validated is not None else validation_payload
                ),
            }
        )
        prompt_compile_rows.append(
            {
                "shot_id": shot_id,
                "shot_prompt": shot_prompt,
                "segment_prompt": segment_prompt,
                "build_ok": bool(prompt_build.get("ok")),
                "build_blocking_issues": list(platform_blockers),
                "recipe": recipe,
                "request": request_payload,
            }
        )

    package: dict[str, Any] = {
        "project_id": project_id,
        "source": {
            "director_run": str(director_path),
            "v2_run": str(v2_path),
            "platform_root": str(platform_path),
        },
        "platform_functions_used": list(PLATFORM_FUNCTIONS),
        "asset_bindings": platform_asset_bindings,
        "platform_contract": {
            "aspect_ratio": aspect_ratio,
            "width": frame_width,
            "height": frame_height,
        },
        "tasks": tasks,
        "input_validation": validation_rows,
        "prompt_compile": prompt_compile_rows,
        "segment_prompt": segment_prompt,
        "build_production_prompt": {
            "ok": bool(prompt_build.get("ok")),
            "blocking_issues": list(prompt_build.get("blocking_issues") or []),
            "advisories": list(prompt_build.get("advisories") or []),
            "asset_ids": list(prompt_build.get("asset_ids") or []),
            "voice_ids": list(prompt_build.get("voice_ids") or []),
            "asset_refs": list(prompt_build.get("asset_refs") or []),
            "renderer": prompt_build.get("renderer"),
        },
        "llm_usage": {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "provider_billed_cost": None,
            "image_generation_called": False,
            "video_generation_called": False,
        },
        "acceptance": {
            "structural": "PASS" if all(row["valid"] for row in validation_rows) else "FAIL",
            "prop_continuity": "NEEDS MANUAL REVIEW",
            "text_continuity": "NEEDS MANUAL REVIEW",
            "timing": "NOT VERIFIED",
            "production_ready": False,
        },
    }
    if output_dir is not None:
        package["_output_paths"] = {
            name: str(path) for name, path in write_h3_production_package(package, output_dir).items()
        }
    return package


def _shot_list_markdown(package: dict[str, Any]) -> str:
    lines = [
        "# H3 Production Shot List",
        "",
        "STRUCTURAL: " + str(package["acceptance"]["structural"]),
        "PROP CONTINUITY: " + str(package["acceptance"]["prop_continuity"]),
        "TEXT CONTINUITY: " + str(package["acceptance"]["text_continuity"]),
        "TIMING: " + str(package["acceptance"]["timing"]),
        "PRODUCTION READY: " + str(package["acceptance"]["production_ready"]).lower(),
        "",
    ]
    for task in package["tasks"]:
        lines.extend(
            [
                f"## {task['shot_id']}",
                "",
                f"- Status: {task['status']}",
                f"- Camera: {task['camera']['camera_id']} "
                f"(parent: {task['camera'].get('parent_camera_id') or '-'})",
                f"- Director duration: {task['director_duration_seconds']}s",
                f"- H3 duration: {task['h3_duration_seconds']}s",
                f"- Timing: {task['timing_status']}",
                f"- Blocked reasons: {', '.join(task['blocked_reasons']) or '-'}",
                "",
                "### Prompt",
                "",
                task["prompt"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_h3_production_package(
    package: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write machine-readable and human-readable H3 package artifacts."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "h3_production_tasks.json": out / "h3_production_tasks.json",
        "h3_prompt_compile.json": out / "h3_prompt_compile.json",
        "h3_asset_bindings.json": out / "h3_asset_bindings.json",
        "h3_input_validation.json": out / "h3_input_validation.json",
        "h3_shot_list.md": out / "h3_shot_list.md",
        "llm_usage.json": out / "llm_usage.json",
    }
    _write_json(paths["h3_production_tasks.json"], package["tasks"])
    _write_json(paths["h3_prompt_compile.json"], package["prompt_compile"])
    _write_json(paths["h3_asset_bindings.json"], package["asset_bindings"])
    _write_json(paths["h3_input_validation.json"], package["input_validation"])
    _write_json(paths["llm_usage.json"], package["llm_usage"])
    paths["h3_shot_list.md"].write_text(_shot_list_markdown(package), encoding="utf-8")
    return paths


__all__ = [
    "compile_h3_production_package",
    "write_h3_production_package",
]
