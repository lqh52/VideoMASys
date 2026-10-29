# Structured v4: fixes and conditional release

Status: implementation and fixed 40-case live evaluation complete; **release rejected**.
The public Space remains unchanged on its existing `clean` profile. The candidate
missed both requested numerical thresholds and the separate library gate.

## Fixed behavior

- Keep the existing SVP-style planner, typed retrieval program, cached scene
  search, evidence inspection, answer, and review roles. Keep the pinned
  `gpt-4.1-mini-2025-04-14` model.
- Derive the answer/option dependency subgraphs and identify unused context.
  Preserve every original claim for review; never invent edges or delete
  conflicting context. Unused context alone is no longer a fatal error.
- Retain strict frame provenance, dependency validity, planned-need coverage,
  and selected-option support. Show the existing repair all missing need IDs,
  reference problems, and selection conflicts together. Preserve the complete
  bounded draft instead of truncating valid JSON to 3,000 characters.
- Provide an inspection-derived checklist of facts, uncertainty, and unresolved
  needs. It is not a declaration that a need has been satisfied. The fictional
  answer example now demonstrates two needs linked through an inference.
- Ask review for direct `supported`/`unsupported` option assessments. The
  controller compares them with the draft, while a separate answer judgment
  checks the selected choice or open-ended response. Review does not overwrite
  the answer or choose a fallback on the controller's behalf.
- Constrain choices, frame IDs, observation IDs, need IDs, and review targets to
  the current request's actual registry. Preserve canonical IDs and raw outputs.
- If an optional review/revision fails, retain only an already validated draft
  as unverified with its outstanding issues. Explicit completion flags prevent
  that retained draft from becoming a successful benchmark result.
- Keep purpose inference distinct from observation. An inferred purpose can be
  supported by visible object-directed activity without claiming an unseen result
  was observed. The neutral worked example is fictional, not a library answer.
- Reject a review that proposes an option its own assessment calls unsupported.
  Use distinctive cached-caption wording for search, without treating captions
  as visual proof. Whole-video evidence needs are independent; compilation must
  honor every declared temporal dependency rather than collapse a multi-hop chain.

OpenAI Docs informed the strict, request-specific schemas and nullable abstention
path. Schema adherence does not establish factual accuracy; see the official
[Structured Outputs guidance](https://developers.openai.com/api/docs/guides/structured-outputs).
No model upgrade, extra agent, or unbounded retry was introduced.

## Validation and release protocol

The new fixed sample has 20 NExT-QA validation questions and 20 questions from
EgoSchema's official public labeled subset, with unique videos and seed
`20260909`. All 40 earlier benchmark videos and known library/development
examples are excluded. Official questions/options are preserved; labels are
scoring-only. The sample is not reshuffled after results are seen.

Release requires all of:

1. At least **36/40 non-abstaining final answers**, with matching completed
   answer/review stages. A retained previous draft does not qualify.
2. At least **28/40 correct final choices**, counting failures and abstentions
   incorrect. Report both datasets separately as well as the aggregate gate.
3. All four library examples have valid media/question/option mappings and
   correct, normally completed live answers using the release candidate.
4. Source hashes, manifests, model version, raw outputs, and usage records
   remain consistent. Historical results are not rewritten.

Review disagreement alone does not disqualify a normally completed answer; it
remains visibly unverified. Review acceptance is not ground truth. A failure of
either numerical threshold prevents publication. No automatic fallback answer
or best-of-several-run scoring is used.

The 40-case run has a $4 runner budget and exactly 40 admitted sessions, with
two workers and fresh caption/vector indexes. Library development has a $1 total
allowance: the first two rounds spent $0.1246036, and the final round is capped at
$0.80. These are caps on these runners, not account-wide
spending limits. Estimated cost uses actual reported tokens and
[official GPT-4.1 mini rates](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

Before publication, verify the exact public target and preserve its current
hardware/sleep settings. The inspected target is `lqh52/videomasys-demo`, on
CPU Basic with hosted inference. Its pre-release source revision was
`3cfdfc4303b6de6a3516990793fdd0befd62cc2a`; source and prompt-profile settings
must both be retained for rollback.

## Offline findings

Focused domain, pipeline, schema callback, image, presentation, retrieval prompt,
accounting, and release-gate checks passed: 212 tests in 3.53 seconds. These are
offline fixtures, not model calls or additional video evaluations.

Replaying the previous batch's 37 last raw answers produced 18 structurally
valid answers and 19 invalid answers. No failed case became valid solely by
removing the orphan check: 15 remaining errors involved need coverage. This is
why v4 includes targeted repair and a multi-need example rather than claiming
that unused-claim handling alone recovers the failed cases. No accuracy was
rescored during this replay.

The static library audit passed for four questions across three videos. The two
NExT-QA examples use reviewed UI paraphrases with unchanged option order; the
EgoSchema examples preserve official text. All videos fully decode and their
decoded pixel hashes match the source videos. Thumbnails and UI loading paths
are valid. Sampled visual inspection found no question/media mismatch; this
does not substitute for the separate live-model check.

Records:

- [New 40-case manifest](../artifacts/benchmark_40_v4_2026-09-08/manifest.json)
- [Independent sample audit](../artifacts/benchmark_40_v4_2026-09-08/sample-audit.json)
- [Old-answer structural replay](../artifacts/structured_v4_development_2026-09-08/replay.json)
- [Library media/annotation audit](../artifacts/library_release_2026-09-08/audit.json)
- [Historical v3 benchmark](benchmark_40_2026-09-08.md)

## Live results

Library development is separate from the held-out denominator. All attempts are
retained; successful examples from different rounds are not combined into a pass.

| Library round | Normally completed | Non-abstaining final answers | Correct / 4 | Outcome |
| --- | ---: | ---: | ---: | --- |
| 1 | 4 | 2 | 1 | Sofa and painting abstained; tree chose the wrong option. |
| 2 | 2 | 2 | 2 | Sofa and painting correct; kitchen and tree stopped before answering. |
| 3, final candidate | 2 | 2 | 2 | Tree and painting correct; sofa tool contract and kitchen repair admission failed. |

Round 1 cost $0.0790316 across 62 HTTP requests; round 2 cost $0.045572 across
52. Round 3 cost $0.051524 across 55 actual HTTP requests; total library spending
was **$0.1761276**. Independent audits reconcile tokens, accounting, raw responses, actual
image/schema payloads, and frozen sources. Round 1 motivated general purpose
calibration and internally consistent review proposals. Round 2 exposed the
planner/compiler dependency mismatch; the final candidate repairs that mismatch
without changing the answer calibration or model. Its final focused checks passed
163 tests, and Ruff passed. Each run binds exact source hashes even though the
development candidates share the `structured-evidence-v4` prompt-version label.

The final library check and fixed 40-case benchmark use the same frozen app.
No benchmark case was retried or replaced after seeing its result.
The library gate failed; the final 40-case result also failed both numerical
thresholds. This candidate was not published.
A repeat read-only Hub check confirmed the public Space is still `RUNNING` at
the original source revision, with `VLM_PROMPT_PROFILE=clean`, CPU Basic, and
172,800-second sleep setting unchanged. All deployed library file bytes match
the validated local library. No remote writes or inference were made by that check.

The kitchen failure was **local request admission**, not a rejected OpenAI call
or an invalid image. Its first answer omitted `environment_context` from answer
coverage. The repair request had this conservative admission bound:
14,628 prompt/system bytes + 6,906 schema bytes + 10,800 image allowance + 1,400
output tokens + 256 overhead = 33,990, exceeding the 32,768 guard by 1,222.
The failed repair made no HTTP request and incurred no provider charge. The
unchanged model has not demonstrated that this repair would otherwise succeed.
The sofa failed two tool-program dependency checks before any answer was produced.
Neither failure should be dismissed as perception noise.

The same guard also rejects some **initial** benchmark Answer requests, not only
repairs. The next correction therefore needs admission-aware construction of both
initial and repaired evidence requests, with an explicit local-budget error.
The native structured path already sends the schema separately; there is no full
schema echo in its text prompt to remove. Compact schema serialization alone saves
only 532 bytes in the kitchen repair and would still exceed the current guard.
Do not fix this by silently dropping evidence IDs, truncating claims, or counting
an earlier choice as the final answer. This diagnosis is recorded without changing
the frozen candidate during its test.

### Fresh 40-case result

Every fixed case was attempted once. No failed case was retried, no question was
replaced, and the candidate sources stayed frozen throughout the run.

| Dataset | Cases | Completed answer/review, including abstentions | Non-abstaining final answers | Correct / all cases |
| --- | ---: | ---: | ---: | ---: |
| NExT-QA | 20 | 14 | 11 (55%) | 10 (50%) |
| EgoSchema | 20 | 6 | 6 (30%) | 6 (30%) |
| Total | 40 | 20 | **17 (42.5%)** | **16 (40%)** |
| Release requirement | 40 | — | **36 (90%)** | **28 (70%)** |

There were 20 incomplete analyses, three completed abstentions, one wrong
non-abstaining final answer, and 16 correct final answers. Four failures were
assigned to the Tool agent and 16 to Answer. Retained drafts after optional-stage
failure remain ineligible. The model accepted the one wrong completed choice;
review acceptance is not a correctness certificate.

Independent failure classification:

- **15 local request-size rejections**, before HTTP: eight on first schema
  attempts and seven on repair attempts following a contract error. All exceeded
  the conservative 32,768 admission bound. None exhausted the dollar budget.
- **Five terminal contract failures** after the bounded repair: four Tool-agent
  failures and one Answer failure.
- **Zero provider transport failures.** The generic displayed `invalid_request`
  message concealed a local application limit, not an OpenAI outage.

The independent scoring audit found no flag, label, or provenance discrepancies.
Diagnostic only: 36 cases actually produced a preliminary visual choice, and 32
of those choices matched the official answer. Of these 32, only 16 survived as
correct final answers; 13 were lost to downstream failures and three became
abstentions. None was changed into a wrong non-abstaining final answer. The other
four preliminary choices were wrong: one remained a wrong final answer and three
stopped downstream. Four cases failed before producing a preliminary choice.
These are intermediate-stage diagnostics, **not** 88.9% final accuracy or an
alternative release score. They indicate substantial orchestration loss without
proving the model's descriptions or explanations are visually grounded.

This is a different cohort from the earlier v3 40-case benchmark, so the raw
accuracy difference does not establish an improvement on the same questions.
The relevant outcome is that this candidate missed the predeclared gates.

The new 40-case run made **571 actual HTTP requests** and used **$0.7308916** in
provider-reported token cost. Including all three library development rounds,
this work used **$0.9070192** (about $0.91), within the separate runner caps.
No outstanding reserved or ambiguous request balance remains in the final summary.
These are API usage calculations, not account-wide invoice or billing-cap claims.
The independent accounting audit matched all 571 HTTP 200 responses to unique
settled request tickets and the usage ledger, with 40 admitted sessions. It also
verified all actual wire/schema/image hashes, 61 frozen source files and snapshots,
87 historical control hashes, and the unchanged manifest. The 61 shared candidate
files exactly match the final library run. Audit success validates these records;
it does not turn the failed performance gate into a pass.

Median cold-run elapsed times were 41.1 seconds for NExT-QA and 58.8 seconds for
EgoSchema; corresponding p95 values were 49.7 and 75.5 seconds. These include
failed runs and are not successful-answer latency guarantees. Indexing and
answer-only timings are separately retained in the summary.

- [Fresh 40-case results](../artifacts/benchmark_40_v4_2026-09-08/summary.json)
- [Frozen run configuration](../artifacts/benchmark_40_v4_2026-09-08/run-config.json)
- [Independent fresh-run accounting and source audit](../artifacts/benchmark_40_v4_2026-09-08/cost-audit.json)
- [Independent scoring and final-answer audit](../artifacts/benchmark_40_v4_2026-09-08/result-audit.json)
- [Library round 1 results](../artifacts/library_v4_2026-09-08/summary.json)
- [Library round 1 audit](../artifacts/library_v4_2026-09-08/live-audit.json)
- [Library round 2 results](../artifacts/library_v4_round2_2026-09-08/summary.json)
- [Library round 2 audit](../artifacts/library_v4_round2_2026-09-08/live-audit.json)
- [Final library results](../artifacts/library_v4_round3_2026-09-08/summary.json)
- [Final library accounting audit](../artifacts/library_v4_round3_2026-09-08/cost-audit.json)
- [Final library trace audit](../artifacts/library_v4_round3_2026-09-08/live-audit.json)
- [Read-only library and public-state recheck](../artifacts/library_release_2026-09-08/post-evaluation-readiness.json)
