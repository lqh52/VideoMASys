# Hold-out and open-ended QA — 7 September 2026

This is a four-run diagnostic on two new videos, not a statistically meaningful
benchmark. The runtime and prompts are frozen: no changes or example replacement
in response to the outcomes. These videos are held out from this application's
known development examples, not necessarily from Qwen's pretraining.

## Selection and protocol

- One NExT-QA validation causal question and one labeled EgoSchema question.
- Deterministic selection before reading labels: ascending SHA-256 rank with seed
  `videomasys-holdout-2026-09-07-v1`. Both were the first-ranked eligible item.
- Existing library video IDs were excluded. Neither selected ID appeared in the
  repository's existing source, documentation, or JSON/CSV evaluation files.
- Selection required an available full video, at most 100 MiB and 180.1 seconds.
  No semantic easiness or model-result filter was applied.
- Each original benchmark question is asked without options first. Both
  open-ended runs precede both multiple-choice runs; only caption/vector caches
  are shared. Answers, conversations, and ground-truth labels are not shared with
  subsequent inference calls. Four runs therefore mean two paired questions,
  not four independent questions.
- Multiple-choice grading uses exact reference indices. Open-ended grading is
  manual semantic and visual assessment, not string matching or verifier votes.
  The visual-audit rubric is prepared before reading model outputs.
- Qwen/Qwen3-VL-8B-Instruct BF16 runs through local vLLM, with real MiniLM
  embeddings and the current application pipeline. This is not a hosted
  Transformers/ZeroGPU test. Local inference uses server GPUs, not the browser.
- Reported latency excludes model startup. Open-ended runs build fresh video
  caches; subsequent multiple-choice runs reuse them, so their elapsed times do
  not isolate the effect of answer format.

The [evaluation-only manifest](../artifacts/evaluation/holdout_2026-09-07/cases.json)
records the selection rule, questions, options, and local scorer references. It
is not included in deployment. No benchmark reference answer is sent to a model.

| Dataset / video | Original question | Reference answer |
| --- | --- | --- |
| NExT-QA `2877270840`, 26.05 s | why is the man half naked | feels hot outside |
| EgoSchema `68cd8a82-f3ea-4816-8dca-045616a41f25`, 180 s | Taking into account the majority of the actions performed by c, what can you deduce was the central objective within the garage environment? | The central objective within the garage environment was to fix a bolt. |

The NExT-QA question asks for a subjective/causal explanation: visible shirtless
outdoor activity cannot directly establish that the person feels hot. Label
agreement and strength of visual support will be assessed separately.

## Results

**Multiple-choice: 1/2 matched the benchmark. Open-ended: 0/2 produced a usable
answer.** The pipeline technically completed three runs, but the completed
open-ended garage run returned the literal string `"None"`; that is not a
successful answer. Both multiple-choice answers were accepted by the model
verifier, including the benchmark miss.

| Run, in execution order | Actual output / outcome | Model review | Local time | Repairs |
| --- | --- | --- | ---: | ---: |
| NExT-QA, open-ended | No answer: tool compilation failed twice | Not reached | 32.8 s | 1 |
| EgoSchema, open-ended | `"None"`; complained that no choices were provided | Rejected | 44.9 s | 1 |
| NExT-QA, multiple-choice | “feels hot outside” — reference match | Accepted | 26.7 s | 0 |
| EgoSchema, multiple-choice | “repair a car” — reference is “fix a bolt” | Accepted | 29.5 s | 0 |

Total measured run time was 134.0 seconds, excluding model startup and human
review. Full raw calls, selected frames, repairs, and failures are retained in
[the run directory](../artifacts/evaluation/holdout_2026-09-07/runs/20260907T025728.878987Z/summary.json).
All four cases ran once under the existing two-attempt repair policy. No failed
case was replaced, rerun with a tuned prompt, or omitted.

## What failed, and why

### Open-ended NExT-QA: tool coverage

The planner requested two independent needs: `attire_observation` and
`environment_context`. Both tool-agent attempts attached every locate call and
inspection only to `attire_observation`. The repair added another attire query
instead of covering `environment_context`. The validator rejected both with
`locate calls must cover every evidence need`, so no retrieval or answer occurred.
Queries also drifted toward the glass and drinking actions in the captions rather
than the clothing/environment question. This is a genuine planning-to-tools
reliability failure, not insufficient visual evidence at the answer stage.

### Open-ended EgoSchema: multiple-choice instructions leaked into open mode

Retrieval selected scenes 0, 4, and 6 (0–20, 80–100, and 120–140 seconds), with
observations about grease/fasteners, tools, and brake components. The answerer had
relevant evidence, and correctly skipped the separate `Choice selection` call.
Nevertheless, it produced:

> None

Its limitation was “No answer choices provided for evaluation.” The verifier
also rejected the draft on the premise that choices were required. The failure
was not missing benchmark options: **open-ended mode must not need options**.
The runtime's answer prompt still asks it to audit a candidate (`None` here),
compare supplied choices, and copy chosen option text. The verifier similarly
refers to a selected option. An empty options list and nullable schema are
insufficient when the role instructions remain multiple-choice-oriented.

### Multiple-choice: correctness, specificity, and calibration

The NExT-QA answer matches the reference, but confidently attributes an unobserved
heat motive. The source establishes a bare upper torso and a likely exterior
setting, not temperature or discomfort. Its explanation also uses questionable
claims about direct sunlight and ample space in a close-up. This is label
agreement, not proof of the stated cause.

The EgoSchema answer identifies real automotive work but chooses the broader
“repair a car” option over the annotated “fix a bolt.” Its explanation explicitly
downgrades fixing a bolt as a subtask. This is a benchmark miss, although the
broader objective is visually compatible with the wheel/brake maintenance seen in
the source. It should not be relabeled correct, nor described as a wholly
fabricated activity. A verifier's acceptance does not resolve the benchmark's
preferred level of specificity.

The selected EgoSchema evidence also has a coverage limitation. Scenes 0/4/6
show grease-like material applied to a pin/fastener, insertion/manipulation,
socket/tool preparation, and later assembly handling. The source's clearer
socket engagement around 106–117 seconds and sustained pliers work around
138–177 seconds were largely outside those selected windows. Full-source
confirmation cannot be substituted for what the answerer actually saw.

## Follow-up priorities — not implemented in this evaluation

1. Give open-ended answering and verification their own instructions: answer the
   question from evidence, with no candidate-choice or option-comparison task.
2. Make tool-repair feedback name the missing need IDs and the required coverage;
   inspect whether the generated query actually serves that need as well as
   satisfying its type.
3. Evaluate answer specificity and unsupported explanations separately from
   exact-choice accuracy. Keep calibrated uncertainty for motives not directly
   established by visual evidence.

The deployed app and all production prompts remain unchanged by this evaluation.
The two new videos and scorer references were not uploaded to the Space.

All seven runtime module hashes still match the frozen source. The full test
suite passed **310 tests**, including four new evaluator checks for input/reference
separation, genuine open-ended routing, preservation of failed answer outputs,
and separate MC/open grading. Ruff passed. These engineering tests are not
model-quality results. The temporary local VLM and embedding services were
stopped afterward; existing model weights and caches were retained.

## Reproduction

```bash
.venv/bin/python scripts/evaluate_holdout_qa.py \
  artifacts/evaluation/holdout_2026-09-07/cases.json \
  --source-archive artifacts/evaluation/holdout_2026-09-07/preflight/20260907T025444.230149Z/source_snapshot.tar.gz \
  --output-dir artifacts/evaluation/holdout_2026-09-07/runs
```

The evaluator expects the same local Qwen endpoint at port 8205 and MiniLM
embedding endpoint at port 8206 as the previous audit. The source-video
[pre-result rubric](../artifacts/evaluation/holdout_2026-09-07/visual_audit.md)
was frozen before its reviewers read outputs. It distinguishes semantic alignment
from evidential calibration, and accepts appropriately supported broader
open-ended descriptions rather than requiring exact reference wording.

The subsequent [frame-by-frame result review](../artifacts/evaluation/holdout_2026-09-07/result_review.md)
examined all 30 distinct final selected JPEGs and records per-case rubric scores,
unsupported claims, and citation limitations. The two garage runs used identical
selected frames, making their answer-mode difference particularly clear.
