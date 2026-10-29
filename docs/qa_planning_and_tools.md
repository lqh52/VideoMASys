# QA reasoning and tool execution

This describes the active Gradio pipeline in `zerogpu/core.py`, on dedicated or
ZeroGPU hardware. The separate service prototype has older agent/fallback paths;
those are not the live demo's execution path.

VideoMASys separates **what evidence is needed** from **which operations retrieve
it**. Model roles use Qwen3-VL-8B-Instruct with separate requests.

```text
Question (+ optional choices)
    → Planner: scope, evidence needs, dependencies, stopping condition, budget
    → Tool agent (+ fallible cached timeline): typed locate / relate / inspect program
    → Static validation → real vector retrieval → temporal coverage gate
    → Fresh sampled-frame inspection
    → Choice selection (multiple-choice only) → Answer → Model review
```

A malformed plan/program gets at most one model repair, with the actual response
and validation error retained. A second failure stops the live pipeline; it does
not silently substitute a fabricated or handwritten program. Model review can
request one answer correction using the same evidence, not an unbounded loop.

## In-context examples adapted from StructuredVP

SVP's active NExT-QA prompt files contain 14 paired **Question → Plan** and
**Question + Plan → Program** examples; EgoSchema contains 9. These are fixed
dataset-specific examples embedded in the main prompt files, not similarity-picked
shots. `PromptBundle.load` loads those files; its separately named example fields
are unused placeholders. The iterative-tool variant selects complete named sets.

We reuse the pairing principle, not benchmark examples or unsupported action/object
APIs. [prompts.py](../src/videomasys/zerogpu/prompts.py) contains four independently
authored synthetic pairs, without answer fields, example images, or pretend tool
observations:

| Example | Teaches | In planner | In compiler |
| --- | --- | --- | --- |
| Workbench activity | Independent whole-video coverage | Yes | Whole-video plans |
| Carrying a mat | Same-object outcome after arrival | Yes | Forward dependencies |
| Held helmet | One independent attribute | Yes | No dependencies |
| Trolley stopping | Preceding condition, not later outcome | No; explicit cause rule | Backward dependencies |

The planner receives three balanced demonstrations. The compiler receives one
matching demonstration, or two for mixed backward/forward dependencies. Selection
uses the validated plan, with no extra LLM call or dataset/answer lookup. Examples
and the current task are explicitly separated. Traces record
`prompt_examples.version`, selected IDs, and the full prompt for every attempt.
Tests validate every example using the production schemas and validator.

The planner receives **no current-video captions**: it defines missing evidence
from the question/competing choices. This prevents a fallible cached caption from
being its only basis for predicting an outcome or stopping at mere arrival.
Only the compiler receives cached captions. For global questions the controller
groups those hints into actual beginning/middle/end thirds, so three queries are
not accidentally based on just the first three scene entries.

The captioner, inspector, answerer, choice reader, and reviewer retain their role
instructions/output contracts; demonstrations are not added to image-heavy calls.
Their frame/token budgets are unchanged. More text still adds some prefill cost.
The runtime checks the real token count against 8192; character limits alone
cannot guarantee that arbitrary text fits.

## Typed retrieval

- `locate(query, limit, evidence_need_ids, temporal_position)` searches real cached
  caption embeddings. Every need must have a direct locate, even if it also has a
  temporal dependency. Queries are hypotheses, not verified observations.
- `relate(source, relation, horizon_seconds, limit, evidence_need_ids)` searches
  windows relative to the highest-ranked source hit, reranking by the target
  need's direct locate query. Dependency order is retrieval order, not necessarily
  video chronology: a `before` need can depend on the event it precedes.
- `inspect(sources, evidence_need_ids)` selects deduplicated scene IDs from actual
  candidates under the three-window/60-second budget. Distinct overlapping scene
  IDs are not silently merged or treated as proof of event order.

The validator checks sequential IDs, earlier-source references, need coverage,
budget bounds, unused outputs, and an inspected relation for every dependency.
Missing direct locates are repaired by the model, not silently added by the live
controller. Local-event retrieval retains its existing target-aware ranking.
An anchor-overlapping window is a candidate, not confirmation of ordering. The
horizon measures the boundary gap, not additional inspected seconds.

## Whole-video coverage is not an event chain

An overall-objective question requires three independent context needs in
early/middle/late order, all with `depends_on=[]`. The hard compiler contract is:

```text
l1 = locate(..., temporal_position="beginning", evidence_need_ids=[first_need])
l2 = locate(..., temporal_position="middle",    evidence_need_ids=[second_need])
l3 = locate(..., temporal_position="end",       evidence_need_ids=[third_need])
inspect(sources=["l1", "l2", "l3"], evidence_need_ids=[first_need, second_need, third_need])
```

Exactly those three locates are allowed; `relate=[]` is mandatory. A global scene
belongs to the third containing its **midpoint**, so a straddling window cannot
satisfy two periods. Local positional filters retain window-overlap semantics.

The executor considers combinations of returned top-k hits and selects one
distinct representative per third within three windows/60 seconds. This is a
small bounded search, not another model call or full-video visual pass.
`temporal_coverage` reports representatives, candidate counts, and missing periods.
If complete coverage is impossible among those candidates under the budget, the
trace stays visible and the pipeline stops before answering. Temporal coverage
means sampled representatives, not that every action was observed.

Uploads of at least three seconds prepare at least three windows. Sub-three-second
clips retain local QA and fail explicitly if asked for three distinct global
representatives. Old one/two-window caches miss the existing length check;
unchanged longer-video caption indexes remain reusable.

## Verification

Tests cover valid paired demonstrations, routing/current-task separation, the
user's unnecessary-two-relations pattern, visible repair attempts, midpoint
boundaries, straddlers, top-k/budget failures, joint candidate choice, short-video
indexing, and local dependencies. See [the ICL evaluation report](agent_icl_2026-09-07.md)
for real-model checks.

Neither synthetic contract tests nor a few development questions establish a
benchmark accuracy gain. Captions can be wrong, sparse frames can miss events,
and model review is not ground truth.
