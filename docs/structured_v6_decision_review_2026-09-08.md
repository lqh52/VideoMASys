# Structured v6: answer decisions and targeted review

Local experimental changes only. The public Space, `clean` profile, model
(`gpt-4.1-mini-2025-04-14`), retrieval operators, and transport limits are unchanged.

## What changed

- **Planner coverage is advisory.** A search-plan item need not become an answer
  premise or a manufactured limitation. Available observations and unresolved
  needs remain in the trace; `plan_coverage` reports addressed/unaddressed labels.
  These labels do not claim that evidence is sufficient. Unknown IDs still fail.
- **Choices and conclusions are reviewed together.** The reviewer receives an
  explicit `decision` containing the selected choice, displayed conclusion and
  supporting claim IDs. This is derived from actual model output, not a second
  generated answer. A semantic mismatch must be identified by the model reviewer;
  valid references alone cannot prove that text entails a choice.
- **Choice changes are explicit.** `choice_change_reason` is required when a
  multiple-choice answer differs from the preliminary or prior choice, including
  changes to/from abstention. The trace records the transition and the reviewer
  sees its reason. A nonempty reason is not proof that changing the choice is
  justified. The preliminary choice remains a hypothesis, not a fixed gold label.
- **Positive support cannot consist solely of limitations.** An inferred claim
  cannot obtain positive option support just by inheriting frame citations from
  limitation claims. Direct evidence or an observed/inferred supporting path is
  required. Interpretation of those frames still depends on the model.

## Review policy

| Finding | Behavior |
| --- | --- |
| Invalid references, broken graph, malformed required output | Strict validation and the existing one format repair; no invented links or judgments |
| Main answer unsupported, inconsistent with its conclusion, or a better supported choice | One answer revision with an explicit reason for any choice change |
| Main answer accepted, but its explanation/support wording needs correction | One explanation-only revision; multiple-choice selection locked in schema and runtime |
| Context claim or unchosen-option disagreement without a main-answer problem | Visible advisory note; no extra answer revision |

The model must flag context that genuinely undermines the answer in its main
answer judgment. Context is not hidden or assumed irrelevant merely because it
is outside the answer's dependency chain.

There is still **at most one semantic revision total**, not one per category.
Existing per-call format repairs, request caps and evidence budgets remain.
Open-ended explanation edits must preserve conclusion meaning by instruction;
the next review receives the prior conclusion for comparison. Unlike exact
multiple-choice locking, this semantic equivalence cannot be guaranteed by schema.

`accepted` now means main-answer acceptance, with separate
`explanation_accepted`, `explanation_issues` and `advisories`. A fully checked
badge still requires a completed matching review, main-answer acceptance,
explanation acceptance and a non-abstaining answer. The UI distinguishes answer
concerns, explanation concerns and review notes.

## Failure behavior and auditability

An optional explanation edit cannot erase the exact answer/review pair that
already passed the main-answer judgment. If that edit or its review fails, the
pair is restored unchanged, the failed attempt remains visible, and the output
is marked `retained_reviewed_answer` / `explanation_repair_failed`. It remains
unverified because the explanation concern remains. `reviewed_pair` identifies
the actual source Answer and Verifier trace entries; the controller adds an
explicit failure note rather than inventing a review.

This recovery does **not** apply to an unreviewed answer, failed initial review,
or failed substantive answer revision. Those remain incomplete; the preliminary
choice is never substituted. State is cleared before a new run so old repair
warnings cannot leak into a later answer.

Historical artifacts and scoring are not rewritten. Old evaluators that require
the last attempted stages to be the final pair do not implement this new optional
explanation-recovery policy. New checks must bind to `reviewed_pair`, validate the
matching raw/canonical records, and retain any later failed optional attempt.

## Verification

The affected offline domain, pipeline, retrieval, transport and presentation
tests passed (524 tests in the combined run). Focused new regressions cover
advisory-only review, exact choice locking during explanation repair, legitimate
answer changes, missing change reasons, optional repair failure, retained-pair
identity, planner coverage and stale state. These are behavior checks, not model
accuracy measurements.

Independent review checked the restoration path and identified the stale-state
and explicit-pair-binding requirements, which were implemented.

A bounded two-case downstream replay froze the saved sofa and cat frames,
observations and preliminary choices, and ran only the new answer/review stages.
It is an exposed regression diagnostic, not a new end-to-end benchmark or
deployment gate. Both analyses completed; one returned a correct non-abstaining
answer and one abstained.

| Case | Previous final result with these inputs | v6 downstream replay |
| --- | --- | --- |
| Sofa | Abstention after correct preliminary “Unwrap it” | Still abstained; reviewer accepted the abstention |
| Cat | Wrong choice after correct preliminary “cleaning the toilet bowl” | Preserved “cleaning the toilet bowl”; answer and explanation accepted |

The sofa's first Answer response omitted its change reason; the existing format
repair supplied one. However, that reason discussed adding claims to address
needs, not a clear evidence-based objection to the preliminary choice. The
reviewer still passed it. This demonstrates the remaining distinction between
requiring an explicit reason and determining whether that reason is sound.
There was no automatic promotion or substitution of the benchmark-correct choice.

Neither case needed an explanation-only semantic revision, so the choice-lock
and reviewed-pair restoration paths are verified by controlled regression tests,
not exercised by this live replay. Retrieval, inspection, model choice, and
preliminary selection were not rerun. The two saved preliminary choices are
explicitly marked as injected inputs, not new model outputs.

The replay used **5 HTTP requests and $0.0138112**, with no pending or ambiguous
charges. No further paid reruns or deployment were performed. The remaining
sofa behavior prevents claiming that the correct-answer rejection issue is fully
resolved; the previous end-to-end release gates have not been re-established.

Independent audit passed: saved-input isolation, all five actual HTTP payloads,
raw/canonical answer and review records, exact reviewed-pair bindings, frame
provenance, 64 source/snapshot hashes, 518 historical controls and settled costs.
Audit integrity does not imply answer correctness.

Records: [two-case results](../artifacts/structured_v6_answer_replay_2026-09-08/summary.json),
[independent audit](../artifacts/structured_v6_answer_replay_2026-09-08/live-audit.json).

OpenAI Docs informed the prompt cleanup and separation of schema compliance
from answer quality: [Structured Outputs can still contain mistakes](https://developers.openai.com/api/docs/guides/structured-outputs#handling-mistakes).
No model upgrade was made. Prior results remain in the
[v5 report](structured_v5_fixes_2026-09-08.md).
