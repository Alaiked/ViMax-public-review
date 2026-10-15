# Sanitized V7/V8 Director Comparison

## Decision

`NO MERGE`

V8 reduced coverage from four shots and three cameras to three shots and two
cameras. Camera economy improved, but the candidate introduced regressions that
made it unsuitable for production.

## Confirmed V8 Defects

1. In Shot 1, `motion_desc` performed the second speaker's line before the
   first speaker's line. `audio_desc` and the source script require the opposite
   order.
2. The candidate copied source editorial windows into
   `estimated_duration_seconds`. Those windows were reference beats, not
   validated whole-shot durations.
3. The candidate compressed the dedicated final reaction beat into an already
   overloaded shot.

## Usage Comparison

| Metric | V7 | V8 candidate |
| --- | ---: | ---: |
| LLM calls | 6 | 5 |
| Input tokens | 17,199 | 17,208 |
| Output tokens | 5,486 | 5,126 |
| Total tokens | 27,251 | 22,334 |
| Wall time | 79.5352 s | 169.3578 s |
| Provider billed cost | not returned | not returned |
| Image generation called | false | false |
| Video generation called | false | false |

## Sanitized Shot Comparison

| Shot | V7 | V8 candidate | Result |
| --- | --- | --- | --- |
| 0 | 4.5 s; wide; entrance, walk, sit, reaction recovery, reach | 3.0 s; same action chain | Timing regression |
| 1 | 5.5 s; medium two-shot; Role A line then Role B line; prop move | 5.0 s; medium two-shot; motion reverses line order | Dialogue-order regression |
| 2 | 4.8 s; reused medium setup; second exchange and prop move | 4.0 s; exchange, prop move, and final silence compressed | Timing and reaction regression |
| 3 | 1.0 s; reaction close-up; final silent beat | absent | Dedicated reaction beat lost |

## Independent Status

- Structural contract: reviewed separately from director quality.
- Prop continuity: improved in the candidate text, but not a reason to merge.
- Text continuity: `FAIL` because dialogue order is reversed.
- Timing: `FAIL` because editorial windows were treated as shot durations.
- Director quality: mixed; better camera economy, worse reaction coverage.
- Production ready: `NO`.

No private script text, character names, dialogue, asset identifiers, or
deliverable paths are included in this evidence file.
