# ViMax V9 Director Decision Quality - Sanitized Review Evidence

Status: `CANDIDATE / NOT SIGNED OFF`

This evidence note is sanitized for public review. It omits private scripts,
character names, dialogue, assets, paths, and raw run artifacts. It records the
implementation, aggregate measurements, and failure modes needed to review the
V9 experiment without exposing production material.

Private candidate checkpoint:
`86c4e2738170ba12253b4f1cf32555b93836487d`
(`Add V9 two-stage director decision candidate`).

## Change Boundary

V9 keeps the existing `StoryboardArtist` as the only director agent. It adds
one plain-text Stage A call before the existing structured Stage B call:

- Stage A uses `system_prompt_template_interpret_scene` to produce a compact
  advisory directing brief from the original script and character definitions.
- Stage B uses `system_prompt_template_design_storyboard` and the existing
  `StoryboardResponse` / `PydanticOutputParser` path to produce the final
  `ShotBriefDescription` list.
- The original script and character definitions remain the final authority.
- The Stage A text is passed through `<DIRECTOR_INTERPRETATION>` and cannot add
  events, dialogue, props, locations, or motivations.
- Decomposition, Camera Tree, caching, V2 text bridging, and the H3 text
  compiler were not redesigned.
- The exact Stage A and Stage B prompt text is available in
  `agents/storyboard_artist.py`; the runner persists both to
  `director_prompts.json` and `director_prompts.md`.

No new agent, IR, schema, generation adapter, or production workflow was added.

## Real Run Shape

Two scenes with different dramatic mechanisms were run:

- Scene A: a restrained two-person table scene organized around observation,
  blocked action, repeated testing, and silent residue. A signed-off V7
  planning baseline was available.
- Scene B: a public sports-action scene with a crowd, a failed physical action,
  correction, retry, visible success, and a forward-looking final beat. There
  was no same-conditions V7 baseline, so no fair V7 quality delta is claimed
  for this scene.

Scene A V7/V9 observations:

- V7 used four shots totaling 15.8s.
- V9 used three shots totaling 12.0s.
- The first object-control action improved causally: V9 made the observing
  character see the reaching hand before blocking it and kept the object in
  surface contact during the move.
- The opening compressed an entry, walk, seating, gaze follow, and initial
  reach into 3.0s while the existing text bridge estimated about 3.7s of
  performance load. This is a timing regression.
- V9 merged a second test and the final silent reaction into one 4.0s shot.
  The existing bridge estimated about 7.8s of load. V7's separate reaction
  close-up preserved a distinct silent beat. This is a rhythm regression.

Scene B V9 observations:

- A second attempt was staged before the result was revealed.
- The successful result received its own visual punctuation.
- One reaction was tied to a specific technical correction instead of generic
  praise.
- The final beat remained active by ending on preparation for another attempt.
- A cross-shot dialogue-order reversal survived the local single-shot
  validation and changed the scene-level causal order.
- The existing V2/H3 text bridge blocked on an object handoff: the previous
  state ended with the object on a support surface while the next shot treated
  it as already held, without a pickup transition.
- At least one reused camera index did not describe the same position, lens,
  framing, and movement baseline across all uses.

These failures prevent V9 from being signed off as a director-quality upgrade.

## Existing Text-Chain Regression

No image-generation, video-generation, or paid H3 job was called.

| Scene | Structural | Text continuity | Production ready | Blockers | Advisories |
| --- | --- | --- | --- | ---: | ---: |
| Scene A V9 | `PASS` | `NOT VERIFIED` | `NO` | 0 | 8 |
| Scene B V9 | `FAIL` | `FAIL` | `NO` | 1 | 22 |

Scene A entered the existing text chain without a schema or contract failure.
A structural pass does not establish semantic, spatial, timing, or
cinematographic quality. Scene B produced a real continuity failure and is not
production-ready.

## Aggregate Measurements

All calls used the same provider and model within the experiment. The provider
did not return a billed-cost amount, so no cost is inferred.

| Run | Calls | Input tokens | Output tokens | Total tokens | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scene A V7 baseline | 6 | 17,199 | 5,486 | 27,251 | 79.5352 |
| Scene A V9 | 6 | 16,886 | 6,964 | 26,901 | 133.7539 |
| Scene B V9 failed provider attempt | 8 | 5,936 | 3,890 | 9,826 | 112.6716 |
| Scene B V9 retry | 15 | 38,028 | 12,430 | 56,310 | 458.3490 |
| Scene A V9 text bridge | 1 | 13,396 | 4,438 | 17,834 | 332.5164 |
| Scene B V9 text bridge | 1 | 16,586 | 6,926 | 23,512 | 164.2163 |

V9-only experiment totals, including the failed provider attempt and both text
bridges:

- Calls: 31
- Input tokens: 90,832
- Output tokens: 34,648
- Total tokens: 125,480
- Wall time: about 1,201.5s
- Provider billed cost: `null` / not reported
- `image_generation_called`: `false`
- `video_generation_called`: `false`

## Verification

The final focused private regression run reported:

```text
25 passed in 1.67s
```

The public mirror tests only verify the sanitized prompt, two-call ordering,
schema-contract, and input-context contracts. They cannot verify private V2/H3
integration, semantic continuity, cinematic quality, or production readiness.

## Final Judgment

`V9 IS A CANDIDATE, NOT A VERIFIED DIRECTOR-QUALITY UPGRADE`.

The two-stage method produced useful local improvements, especially in causal
attention and action-specific performance. It also introduced or exposed a
timing regression, a lost reaction beat, a cross-shot dialogue-order failure,
and a production-blocking prop-handoff failure.

No V10 work, new agent architecture, image/video generation call, paid H3 job,
or change to the existing H3 production workflow is included in this review
revision.
