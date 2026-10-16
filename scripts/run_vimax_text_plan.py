from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import UUID

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler

from agents.storyboard_artist import (
    system_prompt_template_decompose_visual_description,
    system_prompt_template_design_storyboard,
    system_prompt_template_interpret_scene,
)
from pipelines.script2video_pipeline import Script2VideoPipeline


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Expected a model or mapping, got {type(value).__name__}")


def _unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _usage_from_mapping(payload: dict[str, Any]) -> dict[str, int]:
    input_tokens = _unique(
        [
            _integer(payload.get("input_tokens")),
            _integer(payload.get("prompt_tokens")),
        ]
    )
    output_tokens = _unique(
        [
            _integer(payload.get("output_tokens")),
            _integer(payload.get("completion_tokens")),
        ]
    )
    total_tokens = _unique(
        [
            _integer(payload.get("total_tokens")),
            sum(input_tokens + output_tokens) if input_tokens and output_tokens else None,
        ]
    )
    return {
        "input_tokens": input_tokens[0] if input_tokens else 0,
        "output_tokens": output_tokens[0] if output_tokens else 0,
        "total_tokens": total_tokens[0] if total_tokens else 0,
    }


def _extract_usage(response: Any) -> dict[str, int]:
    candidates: list[dict[str, Any]] = []

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, dict):
            candidates.append(token_usage)
        candidates.append(llm_output)

    generations = getattr(response, "generations", None) or []
    for generation_group in generations:
        for generation in generation_group:
            message = getattr(generation, "message", None)
            usage_metadata = getattr(message, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                candidates.append(usage_metadata)
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, dict):
                token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
                if isinstance(token_usage, dict):
                    candidates.append(token_usage)
            generation_info = getattr(generation, "generation_info", None)
            if isinstance(generation_info, dict):
                token_usage = generation_info.get("token_usage") or generation_info.get("usage")
                if isinstance(token_usage, dict):
                    candidates.append(token_usage)

    for candidate in candidates:
        usage = _usage_from_mapping(candidate)
        if usage["input_tokens"] or usage["output_tokens"] or usage["total_tokens"]:
            return usage
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class LLMUsageCallback(BaseCallbackHandler):
    """Collect call counts, token usage, and elapsed time without logging secrets."""

    def __init__(self) -> None:
        super().__init__()
        self._started: dict[str, float] = {}
        self._calls: list[dict[str, Any]] = []
        self._sequence = 0

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._started[str(run_id)] = time.perf_counter()

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._started.setdefault(str(run_id), time.perf_counter())

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run_key = str(run_id)
        started = self._started.pop(run_key, time.perf_counter())
        self._sequence += 1
        self._calls.append(
            {
                "call": self._sequence,
                "run_id": run_key,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "usage": _extract_usage(response),
            }
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run_key = str(run_id)
        started = self._started.pop(run_key, time.perf_counter())
        self._sequence += 1
        self._calls.append(
            {
                "call": self._sequence,
                "run_id": run_key,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "error": str(error),
            }
        )

    def summary(self) -> dict[str, Any]:
        input_tokens = sum(call["usage"]["input_tokens"] for call in self._calls)
        output_tokens = sum(call["usage"]["output_tokens"] for call in self._calls)
        total_tokens = sum(call["usage"]["total_tokens"] for call in self._calls)
        return {
            "llm_calls": len(self._calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "calls": list(self._calls),
        }


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>").strip()


def write_director_prompts(output_dir: str | Path) -> tuple[str, str]:
    """Persist the exact text-planning prompts without making an LLM call."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "interpret_scene": system_prompt_template_interpret_scene,
        "design_storyboard": system_prompt_template_design_storyboard,
        "decompose_visual_description": system_prompt_template_decompose_visual_description,
    }
    json_path = directory / "director_prompts.json"
    markdown_path = directory / "director_prompts.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        "\n".join(
            [
                "# ViMax Director Prompts",
                "",
                "## system_prompt_template_interpret_scene",
                "",
                "```text",
                payload["interpret_scene"],
                "```",
                "",
                "## system_prompt_template_design_storyboard",
                "",
                "```text",
                payload["design_storyboard"],
                "```",
                "",
                "## system_prompt_template_decompose_visual_description",
                "",
                "```text",
                payload["decompose_visual_description"],
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(json_path), str(markdown_path)


def write_director_report(
    result: dict[str, Any],
    output_path: str | Path,
    director_interpretation: str = "",
) -> None:
    """Write a human-readable shot table, motion plan, and camera tree."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    characters = [_as_dict(item) for item in result.get("characters", [])]
    storyboard = sorted((_as_dict(item) for item in result.get("storyboard", [])), key=lambda item: item["idx"])
    shot_descriptions = sorted(
        (_as_dict(item) for item in result.get("shot_descriptions", [])),
        key=lambda item: item["idx"],
    )
    variation_by_shot_idx = {
        shot["idx"]: shot.get("variation_type")
        for shot in shot_descriptions
    }
    raw_contexts = result.get("decomposition_contexts", {})
    if isinstance(raw_contexts, dict):
        context_items = raw_contexts.values()
    else:
        context_items = raw_contexts
    contexts_by_shot_idx = {
        int(item["shot_idx"]): item
        for item in context_items
        if isinstance(item, dict) and "shot_idx" in item
    }
    camera_tree = sorted((_as_dict(item) for item in result.get("camera_tree", [])), key=lambda item: item["idx"])

    lines = [
        "# ViMax Director Text Plan",
        "",
        "This report contains structured director planning only. No image or video generation was invoked.",
        "",
        "## Director Interpretation",
        "",
        director_interpretation or "No director interpretation was recorded.",
        "",
        "## Characters",
        "",
    ]
    if characters:
        lines.extend(
            [
                "| Index | Identifier | Static features | Dynamic features |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for character in characters:
            lines.append(
                "| {idx} | {identifier} | {static} | {dynamic} |".format(
                    idx=character.get("idx", ""),
                    identifier=_markdown_cell(character.get("identifier_in_scene", "")),
                    static=_markdown_cell(character.get("static_features", "")),
                    dynamic=_markdown_cell(character.get("dynamic_features", "")),
                )
            )
    else:
        lines.append("No characters were returned.")

    lines.extend(
        [
            "",
            "## Shot Table",
            "",
            "| Shot | Camera | Variation | Visual description | Audio description |",
            "| ---: | ---: | --- | --- | --- |",
        ]
    )
    if storyboard:
        for shot in storyboard:
            variation = variation_by_shot_idx.get(shot.get("idx")) or shot.get("variation_type", "pending")
            lines.append(
                "| {idx} | {cam_idx} | {variation} | {visual} | {audio} |".format(
                    idx=shot.get("idx", ""),
                    cam_idx=shot.get("cam_idx", ""),
                    variation=_markdown_cell(variation),
                    visual=_markdown_cell(shot.get("visual_desc", "")),
                    audio=_markdown_cell(shot.get("audio_desc", "")),
                )
            )
    else:
        lines.append("| - | - | - | No storyboard shots were returned. | - |")

    lines.extend(["", "## Per-Shot Motion Descriptions", ""])
    if shot_descriptions:
        for shot in shot_descriptions:
            timing = contexts_by_shot_idx.get(shot.get("idx"), {})
            duration_seconds = timing.get("duration_seconds")
            duration_status = timing.get("duration_status", "not_provided")
            if duration_seconds is None:
                timing_text = "not provided; timing adequacy is not verified"
            elif duration_status == "director_estimated":
                timing_text = (
                    f"{duration_seconds} seconds (director estimate; "
                    "not a user-specified hard duration; "
                    "timing adequacy is not independently verified)"
                )
            elif duration_status == "provided":
                timing_text = (
                    f"{duration_seconds} seconds (user-provided hard duration; "
                    "timing adequacy is not independently verified)"
                )
            else:
                timing_text = (
                    f"{duration_seconds} seconds (source unspecified; "
                    "timing adequacy is not independently verified)"
                )
            lines.extend(
                [
                    f"### Shot {shot.get('idx', '')}",
                    "",
                    f"- Camera: `{shot.get('cam_idx', '')}`",
                    f"- Variation: `{shot.get('variation_type', '')}`",
                    f"- Variation reason: {shot.get('variation_reason', '')}",
                    f"- First frame: {shot.get('ff_desc', '')}",
                    f"- Motion: {shot.get('motion_desc', '')}",
                    f"- Last frame: {shot.get('lf_desc', '')}",
                    f"- Visible characters in first frame: {shot.get('ff_vis_char_idxs', [])}",
                    f"- Visible characters in last frame: {shot.get('lf_vis_char_idxs', [])}",
                    f"- Audio: {shot.get('audio_desc', '')}",
                    f"- Intended duration: {timing_text}",
                    f"- Duration status: `{duration_status}`",
                    "",
                ]
            )
    else:
        lines.append("No per-shot motion descriptions were returned.")

    lines.extend(
        [
            "",
            "## Camera Tree",
            "",
            "| Camera | Parent camera | Parent shot | Reason |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    if camera_tree:
        for camera in camera_tree:
            lines.append(
                "| {idx} | {parent_cam} | {parent_shot} | {reason} |".format(
                    idx=camera.get("idx", ""),
                    parent_cam="" if camera.get("parent_cam_idx") is None else camera.get("parent_cam_idx"),
                    parent_shot="" if camera.get("parent_shot_idx") is None else camera.get("parent_shot_idx"),
                    reason=_markdown_cell(camera.get("reason", "")),
                )
            )
    else:
        lines.append("| - | - | - | No camera tree was returned. |")

    lines.extend(
        [
            "",
            "## Validation Scope",
            "",
            "- Structural validation: PASS (Pydantic, numbering, references, and camera-tree structure only).",
            "- Semantic continuity: NOT AUTOMATICALLY VERIFIED",
            "- Prop continuity: HUMAN REVIEW REQUIRED",
            "- Spatial continuity: HUMAN REVIEW REQUIRED",
            "- Dialogue and performance sync: HUMAN REVIEW REQUIRED",
            (
                "- Timing validation: DIRECTOR ESTIMATES USED; NOT VERIFIED."
                if any(
                    item.get("duration_status") == "director_estimated"
                    for item in contexts_by_shot_idx.values()
                )
                else "- Timing validation: NOT PROVIDED; timing adequacy is not verified."
            ),
            "- Director aesthetics: HUMAN REVIEW REQUIRED",
            "",
            "## H3 Boundary",
            "",
            "ViMax produced text planning artifacts only. MiniMax H3 Ref2VA asset binding and prompt compilation remain in the existing production workflow.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _prepare_fresh_working_dir(working_dir: Path) -> None:
    cached_paths = [
        working_dir / "characters.json",
        working_dir / "storyboard.json",
        working_dir / "camera_tree.json",
        working_dir / "shots",
    ]
    existing = [path for path in cached_paths if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            "Refusing a cached text-plan run. Use a fresh working directory. "
            f"Existing artifacts: {rendered}"
        )


async def run_text_plan(
    *,
    script: str,
    user_requirement: str,
    style: str,
    characters: list[dict[str, Any]] | None = None,
    working_dir: str | Path,
    model_config: dict[str, Any],
    model_factory: Callable[..., Any] = init_chat_model,
    pipeline_factory: Callable[..., Any] = Script2VideoPipeline,
    callback: LLMUsageCallback | None = None,
    allow_cached_artifacts: bool = False,
    shot_decomposition_concurrency: int | None = None,
) -> dict[str, Any]:
    """Run only ViMax's text planning path and persist director-facing outputs."""
    working_path = Path(working_dir).resolve()
    working_path.mkdir(parents=True, exist_ok=True)
    if not allow_cached_artifacts:
        _prepare_fresh_working_dir(working_path)

    usage_callback = callback or LLMUsageCallback()
    factory_config = dict(model_config)
    factory_config.setdefault("callbacks", [usage_callback])
    model = model_factory(**factory_config)
    pipeline = pipeline_factory(
        chat_model=model,
        image_generator=None,
        video_generator=None,
        working_dir=str(working_path),
    )

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        result = await pipeline.plan_text_artifacts(
            script=script,
            user_requirement=user_requirement,
            style=style,
            characters=characters,
            quiet=True,
            shot_decomposition_concurrency=shot_decomposition_concurrency,
        )
    except Exception as exc:
        usage = usage_callback.summary()
        usage.update(
            {
                "started_at_utc": started_at.isoformat(),
                "wall_time_seconds": round(time.perf_counter() - started, 4),
                "model": model_config.get("model"),
                "provider_base_url_host": urlparse(str(model_config.get("base_url", ""))).netloc,
                "provider_billed_cost": None,
                "cost_note": "The provider did not return a billed-cost amount; no cost is inferred.",
                "image_generation_called": False,
                "video_generation_called": False,
                "error": str(exc),
            }
        )
        (working_path / "llm_usage.json").write_text(
            json.dumps(usage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    elapsed_seconds = time.perf_counter() - started

    report_path = working_path / "director_output.md"
    usage_path = working_path / "llm_usage.json"
    contexts_path = working_path / "decomposition_contexts.json"
    interpretation_path = working_path / "director_interpretation.txt"
    revision_path = working_path / "storyboard_revision.json"
    contexts = result.get("decomposition_contexts", [])
    if isinstance(contexts, dict):
        contexts = [contexts[idx] for idx in sorted(contexts)]
    contexts_path.write_text(
        json.dumps(contexts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prompts_json_path, prompts_markdown_path = write_director_prompts(working_path)
    director_interpretation = str(
        getattr(
            getattr(pipeline, "storyboard_artist", None),
            "last_director_interpretation",
            "",
        )
        or ""
    ).strip()
    interpretation_path.write_text(director_interpretation, encoding="utf-8")
    storyboard_artist = getattr(pipeline, "storyboard_artist", None)
    revision_issue = str(
        getattr(storyboard_artist, "last_storyboard_revision_issue", "") or ""
    ).strip()
    if revision_issue:
        revision_path.write_text(
            json.dumps(
                {
                    "issue": revision_issue,
                    "before": [
                        _as_dict(shot)
                        for shot in getattr(
                            storyboard_artist,
                            "last_storyboard_before_revision",
                            [],
                        )
                    ],
                    "after": [
                        _as_dict(shot)
                        for shot in getattr(
                            storyboard_artist,
                            "last_storyboard_after_revision",
                            [],
                        )
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    write_director_report(
        result,
        report_path,
        director_interpretation=director_interpretation,
    )

    usage = usage_callback.summary()
    usage.update(
        {
            "started_at_utc": started_at.isoformat(),
            "wall_time_seconds": round(elapsed_seconds, 4),
            "model": model_config.get("model"),
            "provider_base_url_host": urlparse(str(model_config.get("base_url", ""))).netloc,
            "provider_billed_cost": None,
            "cost_note": "The provider did not return a billed-cost amount; no cost is inferred.",
            "image_generation_called": False,
            "video_generation_called": False,
        }
    )
    usage_path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        **result,
        "pipeline": pipeline,
        "usage": usage,
        "director_output_path": str(report_path),
        "director_interpretation_path": str(interpretation_path),
        "storyboard_revision_path": (
            str(revision_path) if revision_path.exists() else ""
        ),
        "llm_usage_path": str(usage_path),
        "decomposition_contexts_path": str(contexts_path),
        "director_prompts_json_path": prompts_json_path,
        "director_prompts_markdown_path": prompts_markdown_path,
    }


def _load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _model_config_from_environment() -> dict[str, Any]:
    model = os.environ.get("VIMAX_LLM_MODEL") or os.environ.get("AGENT_LLM_MODEL")
    api_key = os.environ.get("VIMAX_LLM_API_KEY") or os.environ.get("AGENT_LLM_API_KEY")
    base_url = os.environ.get("VIMAX_LLM_BASE_URL") or os.environ.get("AGENT_LLM_BASE_URL")
    if not model:
        raise RuntimeError("Set VIMAX_LLM_MODEL or AGENT_LLM_MODEL.")
    if not api_key:
        raise RuntimeError("Set VIMAX_LLM_API_KEY or AGENT_LLM_API_KEY.")
    if not base_url:
        raise RuntimeError("Set VIMAX_LLM_BASE_URL or AGENT_LLM_BASE_URL.")

    config: dict[str, Any] = {
        "model": model,
        "model_provider": os.environ.get("VIMAX_LLM_MODEL_PROVIDER", "openai"),
        "api_key": api_key,
        "base_url": base_url,
        "timeout": float(os.environ.get("VIMAX_LLM_REQUEST_TIMEOUT_SECONDS", "300")),
        "max_retries": 0,
        "temperature": float(os.environ.get("VIMAX_LLM_TEMPERATURE", "0")),
    }
    max_tokens = os.environ.get("VIMAX_NARRATIVE_MAX_TOKENS") or os.environ.get("AGENT_LLM_MAX_TOKENS")
    if max_tokens:
        config["max_completion_tokens"] = int(max_tokens)
    thinking_disabled = os.environ.get("AGENT_LLM_THINKING_DISABLED", "").strip().lower()
    if thinking_disabled in {"1", "true", "yes", "on"}:
        config["extra_body"] = {"enable_thinking": False}
    return config


def _optional_positive_int_environment(name: str) -> int | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer when set.") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be at least 1 when set.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ViMax's script-to-storyboard text planning path only.")
    parser.add_argument("--script-file", required=True, help="Existing script file. The script is not rewritten.")
    parser.add_argument(
        "--characters-file",
        default="",
        help="Optional existing characters JSON to reuse instead of re-extracting characters.",
    )
    parser.add_argument("--user-requirement", default="")
    parser.add_argument("--style", default="Cinematic, coherent, 16:9")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--env-file", default="")
    parser.add_argument(
        "--resume-existing-artifacts",
        action="store_true",
        help="Resume an interrupted run from ViMax's existing validated caches.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.env_file:
        _load_env_file(args.env_file)

    script_path = Path(args.script_file)
    script = script_path.read_text(encoding="utf-8")
    characters = None
    if args.characters_file:
        characters = json.loads(
            Path(args.characters_file).read_text(encoding="utf-8")
        )
    working_path = Path(args.working_dir).resolve()
    working_path.mkdir(parents=True, exist_ok=True)
    (working_path / "script.txt").write_text(script, encoding="utf-8")

    result = asyncio.run(
        run_text_plan(
            script=script,
            user_requirement=args.user_requirement,
            style=args.style,
            characters=characters,
            working_dir=working_path,
            model_config=_model_config_from_environment(),
            allow_cached_artifacts=args.resume_existing_artifacts,
            shot_decomposition_concurrency=_optional_positive_int_environment(
                "VIMAX_LLM_MAX_CONCURRENCY"
            ),
        )
    )
    print(
        json.dumps(
            {
                "director_output": result["director_output_path"],
                "llm_usage": result["llm_usage_path"],
                "llm_calls": result["usage"]["llm_calls"],
                "wall_time_seconds": result["usage"]["wall_time_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
