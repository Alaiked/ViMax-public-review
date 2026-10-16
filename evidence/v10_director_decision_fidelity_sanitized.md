# V10 Director Decision Fidelity: Sanitized Review Evidence

Status: **V10-A CANDIDATE / V10-B NO MERGE / NOT PRODUCTION APPROVED**

This evidence covers the private ViMax V10 experiment from V9 checkpoint
`86c4e2738170ba12253b4f1cf32555b93836487d`. The private source candidate
checkpoints are:

- V10-A: `3fe8276c725a13475803cdb5bcdddc8214818658`
- V10-B: `d1ae11bc0b3ffb29a528abecbee0f24de6c50a45`

V10 changed only the existing `StoryboardArtist` planning path and the existing
text-plan runner diagnostics. It did not add an Agent, IR, Schema, workflow, or
H3 generation component. `ShotBriefDescription`, FF/Motion/LF, Camera Tree, V2,
and H3 contracts were preserved.

## Source Change

The reviewed private source patch is limited to:

| File | Change |
| --- | --- |
| `agents/storyboard_artist.py` | Reorganized Stage B around script timeline, beat-to-shot decisions, performance budget, physical/spatial continuity, reusable camera setups, and a targeted dialogue-fidelity board revision path. |
| `scripts/run_vimax_text_plan.py` | Records the director interpretation, prompt set, dialogue-revision issue, before/after board, and related artifact paths. |

Patch statistics: 2 files changed, 368 insertions, 7 deletions.

The public mirror tests use synthetic `Role A` / `Role B` fixtures. They verify
the V10 Stage B prompt contract and that a missing scripted line triggers a
targeted full-board revision rather than being silently accepted.

## Real Runs

All runs were text-only. No image generation, video generation, paid render,
or H3 production job was submitted. Provider-billed cost was not exposed by
the endpoint and remains `null`; no cost estimate is presented as a bill.

### Scene 1: dialogue and table business

- V10-B output: 3 shots, 12.0 seconds.
- Calls: 7.
- Tokens: 30,161.
- Wall time: 181.4252 seconds.
- Existing V2 text continuity result: 0 blockers, 9 advisories.
- Failure: shot durations did not contain the planned performance:
  approximately 5.1 seconds planned into a 3.0-second shot, 7.7 into 5.0,
  and 7.7 into 4.0.

The scene preserved the required entrances, movement, repeated object
business, and final reaction at the storyboard level, but the performance
duration fit failed. It is therefore not accepted as a director-quality
upgrade.

### Scene 2: sports-hall ensemble action

V10-A first run:

- 9 shots, 36.5 seconds.
- 12 calls, 48,489 tokens, 267.7835 seconds.
- Existing V2 result: 2 blockers, 17 advisories.
- Failure: the first coaching line was omitted, changing the scripted order
  and dependency of encouragement versus the first attempted action.

V10-B completed run:

- 7 shots, 41.0 seconds.
- 13 calls, 51,690 tokens, 358.4427 seconds.
- Existing V2 result: 2 blockers, 15 advisories.
- The omitted line was restored only after a full storyboard regeneration, not
  by a reliable local correction.
- Remaining blockers included a skipped object pickup and an invalid
  body-motion constraint.

Two targeted repair experiments failed:

| Repair run | Calls | Tokens | Wall time | Result |
| --- | ---: | ---: | ---: | --- |
| fix1 | 6 | 26,440 | 322.6939 s | Failed; did not pass the intended correction gate. |
| fix2 | 12 | 54,840 | 392.4081 s | Failed after revision attempts. |

### New first-test scene

This scene was used once as a first-generation test and was not tuned or rerun
after seeing the result.

- 3 shots, 13.0 seconds.
- 6 calls, 22,064 tokens, 203.243 seconds.
- Existing V2 result: 0 blockers, 7 advisories.
- Failure: the stated durations still did not contain the planned
  performance: approximately 7.2 seconds in 4.5, 11.9 in 4.2, and 9.1 in 4.3.

## Independent Findings

The experiment did not validate a quality improvement:

- V10-A improved the Stage B decision organization but could still omit or
  reorder a required scripted line during generation.
- V10-B's full-board revision could restore a line, but did so at high token
  and wall-time cost and did not reliably repair physical state or timing.
- Several scenes still contained performance-time overcommitment. A structural
  pass is not a timing pass.
- The sports-hall run retained cross-shot object pickup and body-motion
  blockers.

Therefore V10 is **not merged** and no director-quality upgrade is claimed.
The correct next decision is independent review of the source, prompts, and
real output before any V11 experiment.

## V2 and H3 Text Regression

The existing V2 text continuity checks ran on the generated boards. Their
blocker/advisory counts are reported above; they do not validate director
quality or physical feasibility by themselves.

H3 validation was limited to text and contract checks. The available local
environments were incomplete:

- One environment failed at import with `ModuleNotFoundError: fastapi`.
- The private platform environment failed at import with
  `ModuleNotFoundError: itsdangerous`.
- Seven H3 compiler tests stopped at import/dependency resolution.
- Four prompt-split tests passed.

No dependency was installed without separate approval. No H3 video or paid
task was run.

## Media Boundary

No image or video generation API was called. No ComfyUI H3 production task was
submitted. This experiment does not claim native ViMax support for the private
H3 workflow and does not modify the existing H3 production workflow.
