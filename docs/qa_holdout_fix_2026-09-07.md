# QA hold-out fixes — 7 September 2026

This follows the [frozen hold-out evaluation](qa_holdout_2026-09-07.md). The
original failed traces and pre-result visual rubric remain unchanged. The same
questions are now regression/development cases, not a fresh unseen evaluation.
Reference answers remain local scorer metadata and are never model inputs.

## Changes

- Dedicated open-ended answer schema containing only answer text, evidence scene
  IDs, inference, and limitations. Choice-selection/comparison instructions are
  absent from the actual open answer request and reviewer input.
- Dedicated open-ended reviewer: evaluate a supported free-form response or a
  genuine visual limitation, never require supplied answer choices.
- Bare placeholders and whitespace do not count as answers. They trigger the
  existing one-repair policy, with the failed raw response preserved. A specific
  uncertainty explanation is valid and is not rewritten into a benchmark label.
- Tool queries must translate the actual need descriptions, not merely repeat
  the most salient cached action. Coverage errors name missing needs and explain
  the required locate-to-inspect connection; existing strict gates remain.
- The concise multiple-choice reader explicitly prefers a directly supported
  specific action over a broad description, without inventing a narrower cause.
- Answer/review rules distinguish missing evidence from contradiction, daylight
  from temperature, one contact from repeated events, and an ongoing action from
  a proved repair outcome. These are model instructions, not guarantees.

No model change, larger video scan, whole-video fallback, or benchmark-derived
answer template was added. Final visual calls remain bounded to
18 images over at most three selected windows spanning 60 seconds. Open-ended
mode still skips the separate choice-selection call.

## Verification plan

Rerun the original four cases with identical questions, options, and order, using
fresh caption caches and a frozen candidate source. Then check all four existing
library multiple-choice examples against the same source snapshot. Old library
caption caches may be reused and will be labeled warm. Evaluate wording and
calibration separately from exact multiple-choice reference agreement.

## First candidate (V1)

All four original cases completed with zero JSON repairs, compared with the
baseline's three completed pipelines and two unusable open-ended responses.

| Case | V1 result | Review | Local time |
| --- | --- | --- | ---: |
| NExT-QA open | Qualified outdoor relaxation/casual-setting explanation | Accepted | 38.9 s |
| EgoSchema open | Brake-system servicing explanation | Accepted | 40.6 s |
| NExT-QA MC | Correct: feels hot outside | Rejected | 26.3 s |
| EgoSchema MC | Broad car repair; reference is fix a bolt | Accepted, benchmark miss | 32.8 s |

The two open answers are substantive, not two exact reference matches. The
outdoor explanation does not specifically infer heat; the brake answer retains
an unverified replacement detail. The [V1 visual review](../artifacts/evaluation/holdout_fix_2026-09-07/v1_review.md)
uses the original frozen rubric and preserves these limitations.

All four library MC answers remained correct. Tree, kitchen, and painting passed
review (29.2, 29.0, and 34.3 seconds). Sofa selected **Unwrap it**, but review
rejected an unsupported claim about removing a toy (32.8 seconds). This surfaced
a correctable draft problem rather than justifying removal of the checker.

## Second candidate (V2): bounded feedback correction

The verifier can now explicitly request one correction with `needs_revision`.
Only a rejected draft with that flag starts the correction: the answerer receives
the specific issues, the previous draft, and the exact same selected frames. It
may remove/qualify unsupported wording or correct citations, but cannot invent
new evidence or silently change the selected multiple-choice candidate. A final
review may still reject it. There is no unbounded loop or forced acceptance.

This introduces **two extra model calls only on the correction path**, retaining
the original and corrected Answer/Verifier entries and labeling the second pair
"Correction 1 of 1" in the UI. JSON validation repair and review-driven correction
are different events and are reported separately. Open-ended mode still skips
choice selection. The full automated suite passed **334 tests** and Ruff.

## V2 real-model results

All four original cases completed with **zero JSON repairs**. Both open-ended
runs produce substantive responses, compared with zero usable answers before
the fix. Neither is proof of perfect factual grounding or a fresh hold-out win.

| Case | Final V2 result | Final review | Time | Review corrections |
| --- | --- | --- | ---: | ---: |
| NExT-QA open | Qualified relaxation/warm-weather explanation; speculative drink identity | Accepted | 29.9 s | 0 |
| EgoSchema open | Brake-system servicing; unverified replacement detail remains | Accepted | 39.7 s | 0 |
| NExT-QA MC | Correct candidate index retained, but corrected prose says heat cannot be established; candidate marked unclear | Rejected | 36.7 s | 1 |
| EgoSchema MC | Broad car repair, not reference fix-a-bolt | Accepted, benchmark miss | 32.7 s | 0 |

The corrected NExT-QA draft is better qualified, but still Needs review. Its
preserved correct candidate index is not equivalent to a supported final answer.
The extra answer/review pair cost **9.8 seconds** in this run. The final reviewer
still requested a revision despite the final-pass instruction; the controller
enforced the one-correction limit and left the rejection visible.

All four library MC regressions matched their references and were accepted, with
zero JSON repairs and zero review corrections:

| Library example | Answer | Warm local time |
| --- | --- | ---: |
| Sofa | Unwrap it | 28.8 s |
| Return to tree | Continue unwrapping presents | 28.6 s |
| Kitchen | C is cleaning dishes | 29.4 s |
| Painting | To clean the paintbrush | 33.5 s |

**Sofa passed on its first answer/review pair.** Therefore this run does not prove
that the correction pass fixed sofa: the changed review prompt/schema also changes
generation behavior. The only exercised correction path was the uncertain heat
question, where final acceptance was not forced.

The same seven-module snapshot was used for both V2 sets, archive SHA-256
`5bbee8f951ecbfa2ecece38ef2bad82082f6b4db192c53dd71fc15a43b4dbfd9`.
Artifacts:

- [Original-case retest](../artifacts/evaluation/holdout_fix_2026-09-07/v2/20260907T032815.677705Z/summary.json)
- [Library regression](../artifacts/evaluation/holdout_fix_2026-09-07/library/improved-holdout-fix-v2/20260907T033052.626566Z/summary.json)
- [Human evidence review](../artifacts/evaluation/holdout_fix_2026-09-07/v2_review.md)

These are local BF16 vLLM measurements, not hosted Transformers/ZeroGPU latency
promises. Indexing is cold for the original-case open runs and warm for the MC
pairs and library. No labels, review rubrics, or answer templates were passed to
the runtime or deployed. Remaining issues include overly specific visual claims,
reviewer inconsistency, and benchmark objective granularity. The original cases
are retained as regression tests, not advertised as new unseen evidence.

## Engineering and deployment

The final full suite passed **335 tests**; Ruff passed. Tests cover choice-free
open schemas/prompts, placeholder rejection and repair, useful uncertainty
responses, missing-need diagnostics, one-correction enforcement, retained
original drafts/reviews, unchanged evidence inputs, and correction labels.
Local evaluation services were stopped after testing; downloaded weights and
artifacts were retained and unrelated GPU services were not touched.

The private ZeroGPU Space was updated to commit
`a054abd2c753fe2d123495639b2b8e6e1ba7b7c5`. The deployment allowlist includes app
source and the existing library only, not hold-out videos, scorer references,
audits, or test fixtures.

The deployed Space reached `RUNNING`, and its Gradio configuration returned HTTP
200. A real hosted open-ended smoke test used the existing kitchen library video
with **no choices and no reference answer supplied**. It completed in **59.2 s**
including cold indexing and hosted execution, with no JSON repairs, no
choice-selection call, and no review correction. The answer described washing
dishes, scrubbing residue, rinsing, and arranging clean items on a drying rack;
the model reviewer accepted it. This is one functional hosted check, not a
latency guarantee or independent proof of correctness.

[Hosted trace](../artifacts/evaluation/holdout_fix_2026-09-07/hosted-open/20260907T033900.637629Z/01-library-kitchen-open.json)
preserves the actual streamed outputs, prompts, model responses, selected
evidence, final answer, and review.
