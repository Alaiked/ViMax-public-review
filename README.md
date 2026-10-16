# ViMax Public Review Mirror

This repository is a sanitized, read-only code-review mirror for the ViMax
director text-planning work described in the private source repository
`Alaiked/ViMax`.

It is not a complete ViMax distribution and it is not production-ready on its
own. The public copy intentionally contains only the director-planning,
First/Motion/Last decomposition, V2 action-bridge, H3 task-compiler, focused
tests, and sanitized review evidence needed to inspect those changes.

## Provenance

- Upstream project: [HKUDS/ViMax](https://github.com/HKUDS/ViMax)
- Upstream baseline: `v1.2.0`
- Upstream baseline commit recorded by the private repository:
  `fd4b72e7731be6e5f88486206d25ea59849b0285`
- Private source repository: `Alaiked/ViMax` (private)
- Private source branch: `master`
- Private V9 base commit:
  `b83c66d9d1437e6882d42a16145cbc6f11eeba6e`
- Private V9 base subject:
  `Sync signed-off ViMax planning validation fixes`
- Private V9 candidate commit:
  `86c4e2738170ba12253b4f1cf32555b93836487d`
- Private V9 candidate subject:
  `Add V9 two-stage director decision candidate`
- The V9 candidate is copied from that signed-off private checkpoint. The
  public review commit SHA is recorded by GitHub after this mirror commit.
- License: the original upstream `LICENSE` is preserved unchanged.

This public mirror uses a new single-commit Git history. It does not publish or
rewrite the private repository's Git history.

## What Is Included

- `agents/storyboard_artist.py`: director storyboard prompt, B-Sync
  decomposition prompt, decomposition context inputs, and duration context.
- `pipelines/script2video_pipeline.py`: the text-planning path and the
  pipeline-side context propagation used by the runner.
- `interfaces/`: the Pydantic contracts needed to inspect `ShotBriefDescription`
  and `ShotDescription`.
- `scripts/run_vimax_text_plan.py`: ViMax text-planning runner.
- `scripts/run_vimax_v2_action_bridge.py`: adapter from ViMax text artifacts to
  the existing private V2 planner and validator.
- `scripts/h3_production_agent.py`: compiler from ViMax/V2 artifacts into
  existing H3 production request shapes. It does not submit jobs.
- `tests/`: focused public tests using synthetic fixtures.
- `evidence/v7_v8_sanitized_comparison.md`: sanitized V7/V8 comparison and the
  V8 `NO MERGE` decision.
- `evidence/v9_director_decision_quality_sanitized.md`: sanitized V9
  two-stage-director implementation evidence, aggregate measurements, and real
  failure cases, with no private script or character material.

## Public-Version Differences From The Private Source

The production logic is intended to remain source-identical. The public mirror
contains only the following deliberate differences:

| Area | Private source | Public mirror | Reason |
| --- | --- | --- | --- |
| V2 root | Hard-coded local Windows path | `VIMAX_V2_ROOT`, defaulting to a relative path | Remove internal absolute path |
| Bridge defaults | Hard-coded private deliverable paths | `VIMAX_SOURCE_RUN` and `VIMAX_BRIDGE_OUTPUT_DIR`, with relative defaults | Remove internal absolute paths |
| Prompt wording | Basketball-specific acquisition/retrieval example | Generic acquisition/retrieval wording | Remove private test-story detail |
| Prompt wording | Occupation-specific accessory examples | Generic role-typical accessory/tool wording | Avoid exposing test-fixture hints |
| `interfaces/scene.py` examples | Named example dialogue | `Role A` / `Role B` and `Speaker A` / `Speaker B` | Generic fixture names |
| Public-only files | Not present | `README.md`, `PROVENANCE.md`, `.gitignore`, `pyproject.toml`, public tests, sanitized evidence | Make the review scope and tests independently readable |

No V2 Validator, object-lifecycle model, Pydantic contract, bridge behavior,
director workflow, or H3 request shape was redesigned for this public copy.

## Dependencies On Private Platform Code

The bridge dynamically imports these existing private V2 modules:

- `agent.schema.storyboard_ir`
- `agent.harness.planners.planner_h`
- `agent.harness.production`
- `agent.llm.openai_compat`

The H3 compiler calls these existing private platform entry points:

- `agent.harness.director.materialize.materialize`
- `agent.harness.production.build_production_prompt`
- `backend.app.h3_recipe.compile_recipe`
- `agent.prompt.request_builder.GenerationRequestBuilder.build`
- `backend.app.routers.jobs.CreateJobRequest.model_validate`

Those platform modules are not published here. Tests for those integrations are
therefore limited to pure helper contracts unless the private platform is
available separately.

The full ViMax text-plan runner also depends on original ViMax modules such as
`tools.render_backend`, `utils.provider_presets`, `utils.image_selection`, and
the complete `agents` package. This mirror does not replace those modules.

## Excluded Material

The public mirror does not contain:

- `scripts/inputs/**`
- original `deliverables/**`
- private scripts or character fixtures
- real character or scene assets
- asset UUIDs or production asset paths
- API keys, tokens, cookies, `.env` files, databases, or credentials
- server configuration or internal service information
- model files, generated images, video, audio, or large binary artifacts
- private Git history

## Validation

Run the focused tests from this directory:

```bash
python -m pytest
```

The public tests are intentionally synthetic. They verify Stage A/Stage B
ordering and authority, prompt and context contracts, require dialogue in
`motion_desc` to preserve the source order, and require incomplete `[Shot N]`
prompt splits to fail instead of duplicating a shot. A structural pass is not a
director-quality pass, a prop-continuity pass, a timing pass, or a
production-ready declaration.

The V8 external-director candidate is retained as `NO MERGE`: it improved camera
economy but reversed two dialogue lines in `motion_desc` relative to
`audio_desc`, copied editorial timing windows into shot durations, and removed
the dedicated final reaction beat.

The V9 candidate is not signed off. It improves some causal attention and
action-specific performance choices, but the real runs exposed a timing
regression, a lost reaction beat, a cross-shot dialogue-order failure, and a
prop-handoff blocker in the existing text-continuity validation.

## Scope Boundary

This mirror does not call image generation, video generation, MiniMax H3, or any
paid rendering service. It does not modify the private V2/H3 production
workflow and does not claim native ViMax support for the private H3 workflow.
