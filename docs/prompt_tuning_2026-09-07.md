# Grounding-focused prompt tuning — 7 September 2026

## Protocol, fixed before new model calls

Target: `gpt-4.1-mini-2025-04-14`, temperature 0, no reasoning step.
Baseline: unchanged `clean-roles-v2`. Candidate: `grounded-roles-v1`, selected
with `VLM_PROMPT_PROFILE=tuned`. The baseline source remains available verbatim.

The new owner-requested tuning batch has a separate local ledger allowing at most
60 HTTP attempts, including failed requests and retries. Previous experiment
ledgers and the public app's allowances are not reset or increased. This is a
request cap, not an account billing cap.

The OpenAI Docs guidance favors clear rules, targeted examples for persistent
failures, and empirical evaluation. Here that means a short, unrelated example
distinguishing missing evidence from contrary evidence—not benchmark answers or
long reasoning instructions. [Official GPT-4.1 prompting guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-4.1).

### Controlled stage replay

Replay the exact image bytes, frame order, timestamps, question, choice candidate,
schema and original fallible inspection notes from the two existing Mini runs:

- NExT-QA sofa purpose, multiple choice: inspection scene 3, answer, review of
  the original flawed draft, and review of the newly generated answer.
- EgoSchema kitchen objective, open-ended: inspection scene 6 and the same
  answer/review comparisons.

Both profiles receive the same inputs. Fresh inspector outputs are evaluated
separately and are **not** substituted into the answer replay. The tuned reviewer
deliberately omits inspector descriptions from its prompt; this is an explicit
part of the candidate, not hidden extra visual evidence. Reviewing each profile's
new answer is diagnostic, while reviewing the same original draft is controlled.

Prompt changes affect inspection, refinement, answering and verification only.
Captioning, planner, tool compiler, choice selection, schemas, retrieval,
embeddings, frame sampler, token ceilings and maximum correction count are
unchanged. The candidate adds no model call to the pipeline.

### Acceptance criteria

Score main-answer correctness separately from factual grounding and verifier votes.

- Keep positive evidence: clearly visible physical unpacking supports that action;
  repeated visible cleaning supports a cleaning objective.
- Do not invent a pickup or arrival in a scene that starts with the actor already
  working on the object.
- Avoid unsupported tool identities and claims that a process visibly achieved
  its goal without a paired before/after state.
- Label an unobserved alternative unclear, not contradicted, unless an actual
  incompatible fact is shown.
- Do not claim a person was absent when that person is visibly interacting with
  the object. Check limitations and inference as well as the main answer.
- Cite the scenes that actually support each factual claim.
- The reviewer should reject material false claims but accept grounded purpose
  inferences; making every answer uncertain is not an improvement.

Two full-pipeline validation cases are selected before inspecting candidate
results: a fresh NExT-QA ball question from a video absent from prior development
artifacts, and the painting-library question in open-ended mode. The latter is
held out from this tuning pair, not from the project's historical evaluations.
Reference answers remain scorer-only. These cases check generalization; they
are not a statistical benchmark or evidence about model pretraining.

## First candidate: not suitable for promotion

All four replays completed with four calls each. The replayed original flawed
draft was accepted in both cases by both profiles: **0/2 detected** for clean-v2
and grounded-v1. Both new sofa answers still label all four alternatives
contradicted without sufficient incompatible evidence. Both kitchen answers
still claim visibly improved cleanliness without supported paired before/after
states. Grounded-v1 also misnames the blue pad as a brush in inspection.

The independent audit scored the two executed inspection probes at 4/8 checks
for clean-v2 and 5/8 for grounded-v1. The small gain reflects better recognition
of unpacked contents, not removal of invented temporal events or a reliable
reviewer. All four headline answers remain correct. These results do not
justify deploying grounded-v1.

The frozen [visual rubric](../artifacts/mini_tune_2026-09-07/pre_tuning_review.md)
records the two selected inspection probes; the original three-probe proposal
was narrowed before outputs. Sofa arrival is checked in full-answer evidence,
not counted as an independently replayed inspection probe.

## Second candidate protocol

`grounded-roles-v2` is one additional attempt under the **same** 60-attempt,
8-session ledger. V1 source is retained in `tuned_prompts_v1.py`; each run also
contains its actual prompt source snapshot. Baseline requests are not rerun.
Version-specific artifact directories prevent overwriting v1 results.

V2 moves instructions after task data, puts the review checklist after the
actual draft, and uses short JSON examples for missing-evidence classification
and rejection. The examples contain no benchmark objects or answers. The
answerer now independently uses the same images without inheriting inspector
descriptions. Inspection still determines the bounded refinement request and
remains visible in the trace. This omission is a deliberate prompt-input
treatment, not a claim of comparing wording alone. Frame bytes, schema,
candidate, retrieval and generation parameters remain fixed.

The same two development probes are replayed. Only after freezing v2 are the
two preselected full-pipeline validation cases run. No prompt changes in
response to held-out outcomes are permitted in this batch.

## Final results and decision

**Keep the current clean prompts live. Neither candidate established a sufficiently
consistent grounding improvement to promote.** No Space files, variables,
hardware, sleep setting, visibility, credentials or usage allowances were changed.

The second candidate produces more useful citations on the sofa question and
detects genuine problems in the original kitchen draft. However, its inspector
still misidentifies the blue pad and overclaims cleaning success. Its reviewer
sometimes rejects visible positive evidence while endorsing unsupported
alternative exclusions. The trace retains these errors; no answer or review was
silently replaced.

| Fixed-frame diagnostic | Clean-v2 | Grounded-v1 | Grounded-v2 |
| --- | ---: | ---: | ---: |
| Correct main conclusion, two development questions | 2/2 | 2/2 | 2/2 |
| Inspection checks passed, two windows × four checks | 4/8 | 5/8 | 6/8 |
| Original flawed drafts rejected, regardless of reason | 0/2 | 0/2 | 2/2 |
| Original flawed drafts with a valid detected issue | 0/2 | 0/2 | 1/2 |

These are descriptive human-audit checks on two development examples, not
benchmark accuracy percentages. V2's sofa rejection does **not** earn detection
credit: it wrongly denies visible unpacking and endorses the draft's false claim
that nobody else was nearby. Its kitchen rejection correctly identifies the
unsupported cleanliness claim and wrong tool name, but also contains confused
feedback. The new kitchen draft's review wrongly denies visible running water.

Both cases have identical recompressed image hashes and strict-schema hashes
across all three profiles. The original clean prompt/schema/image-name replay
checks also pass. Thus these are genuine fixed-frame prompt treatments; the
answer/inspector-note omission in v2 remains an explicit part of that treatment.
Model temperature zero does not make separate runs a statistical guarantee.

### Full-pipeline validation of frozen v2

| Question | Main answer | Evidence/review outcome | Time | HTTP calls | Estimated USD |
| --- | --- | --- | ---: | ---: | ---: |
| Fresh NExT-QA `2535384528`, multiple choice | `walks away`, exact reference match | Correct late action, but three unsupported contradiction labels remain. Reviewer rejects twice with excessive requirements about ball position/gaze. | 45.664 s | 14 | 0.0208500 |
| Painting library, open-ended | Cup used to rinse/clean the brush | Plausible main function, but claims proven color separation/clarity and cleanliness beyond sampled evidence. Reviewer accepts. | 56.085 s | 17 | 0.0181900 |

The ball run retrieved all three windows, including the outward look and later
walking. It is not a retrieval coverage failure. Final citation `[2]` supports
walking but omits the earlier observation/swing context used in its explanation.
The model reviewer demands more than the task needs, while missing real errors
in option-comparison statuses. It requested one correction; the final rejection
is preserved rather than retried until acceptance.

The painting run retrieved windows 1/5/8 and refined a brief brush/cup interaction
around 28 seconds. Those images support a brush/water interaction in a painting
workflow, not a demonstrated causal improvement in color quality. Full-source
knowledge or an expected answer is not substituted for selected-frame evidence.
The independent audit confirms cup contact at approximately 28.63 and 179.75
seconds; the inspector wrongly places the late contact at 171.95 and then denies
the actual last-frame contact. Under the frozen rubric, painting semantic
adequacy is 1/2 (correct role, inadequately qualified contribution) and calibration
0/2 (unproven color-quality effect). Ball grounding is 1/2 despite an exact-choice
match. These scores keep a useful core answer separate from unreliable details.

Both full runs used fresh caption indexes and one bounded refinement. Neither had
schema repairs, HTTP retries or reasoning tokens. There is no clean full-run
comparison on these exact validation inputs, so their latencies cannot establish
a prompt-related speedup or slowdown.

### Accounting, artifacts and engineering checks

Eight admitted sessions used **55/60 HTTP attempts**, including 24 development
replay calls and 31 full-pipeline calls. Estimated cost from reported usage is
**USD 0.0792792**, using fixed standard Mini rates of USD 0.40 input, 0.10 cached
input and 1.60 output per million tokens. This is an estimate, not an invoice or
account spending cap. All prior ledgers remain untouched; this batch's eight
session allowance is consumed and was not reset.

Artifacts are under [the experiment directory](../artifacts/prompt_tuning_2026-09-07/).
Each run includes raw prompts/responses, strict schemas, source snapshots/hashes,
image provenance, usage, completion status and review results. A report's
`result=passed` means execution completed, **not** that grounding passed.
Independent claim-by-claim audits cover [clean/v1 replay](../artifacts/prompt_tuning_2026-09-07/replay_visual_review_v1.md),
[v2 replay](../artifacts/prompt_tuning_2026-09-07/replay_visual_review_v2.md) and
[full validation](../artifacts/prompt_tuning_2026-09-07/full_validation_visual_review.md).

The reusable [runner](../scripts/evaluate_prompt_tuning.py) requires explicit
`--run`, fixes the model and ledger, rejects unsupported modes/versions, and
refuses existing output directories. It does not upload held-out videos or scorer
labels. The original clean source hash remains
`4ef949bba3b0cf132d84e8dd3d508726d1cd9e3360da7dbf178e8a0f21c7fdb7`.

Full suite: **825 tests passed**, Ruff clean. These are software-contract tests,
not additional live model-quality results. The public management API confirms
the existing commit `3cfdfc4303b6de6a3516990793fdd0befd62cc2a` is still RUNNING,
public, CPU Basic, sleep 172800 seconds, model `gpt-4.1-mini-2025-04-14`, profile
`clean`. No new deployment was made. Browser inference was not re-tested because
the workspace's network filter still blocks the Space domain; the live model
tests above used the same local pipeline with the owner's direct API key.

### What to test next

More instructions alone have not reliably resolved these errors. A next bounded
experiment should test timestamp-linked factual claims and explicit visual
discriminators as the evidence contract, with semantic consistency checks before
claiming a draft is model-checked. That is a proposed next experiment, not a
proven fix, and is not implemented or charged for in this batch.
