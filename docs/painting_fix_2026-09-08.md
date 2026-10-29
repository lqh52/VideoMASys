# Painting failure: diagnosis and bounded v7 fix

This experiment kept the public baseline unchanged. The user subsequently
approved publishing v7; see the [deployment record](public_v7_deployment_2026-09-08.md).

The saved full GPT-5.4 v6 run selected “To provide a source of water for the
paintbrush.” The official EgoSchema answer is “To clean the paintbrush.” Review
accepted the wrong choice; it did not reject or change a correct Answer draft.

## What went wrong

1. **Scope repair lost a real dependency.** The original plan combined
   `whole_video` with an event-relative “after cup contact” need. The independence
   guard's repair message told the planner to remove dependencies instead of
   considering `local_event`. The repaired plan then led to an overview-based
   search restricted to scenes 0, 3 and 6 (0–20, 60–80 and 120–140 seconds).
2. **A useful cache hint was truncated.** Scene 0's caption described reaching
   toward the cup, but that sentence began at character 263. The tool agent saw
   only the first 240 characters. Full captions were embedded, so this was a
   compiler-context issue, not lost vector-index text.
3. **Inspection missed contact already in its evidence.** Visual examination of
   the original video shows brush/cup contact around 19–20 seconds. The selected
   19.75-second frame already contains this interaction, yet inspection reported
   no contact and requested no refinement there. Another contact around
   52.5–53.5 seconds fell outside the selected windows. The actual close look at
   73.85–77.85 seconds examined palette activity rather than cup contact.
4. **The alternatives were judged inconsistently.** Answer and Review inferred
   “water source” from the painting setup but demanded visible interaction for
   “clean the paintbrush.” Context alone supported neither distinction reliably.

Repeated dipping between painting and palette work supports a cleaning/rinsing
interpretation. The frames do not clearly establish pigment removal or prove
that cleaning is the cup's exclusive possible function. That remains an inference,
not a newly asserted visual fact.

Original evidence: [v6 bundle](../artifacts/gpt54_library_2026-09-08/cases/egoschema-painting/bundle.json)
and [raw trace](../artifacts/gpt54_library_2026-09-08/cases/egoschema-painting/raw-trace.json).

## Changes

- Clarify that a question about an object's use is event-focused even if it says
  “overall process.” Event-focused retrieval can search repeated occurrences
  anywhere in the video. Whole-video scope remains for overall activity questions.
- Let planner repair correct scope while preserving genuine temporal edges;
  remove only conceptual edges when the plan truly needs whole-video coverage.
- Give only the structured compiler complete captions within their existing
  600-character schema bound. Other profiles retain their 240-character hints.
- Direct inspection to tool tips/contact points and use the existing close-look
  request for an important interaction ambiguous in even one sampled frame.
- Share one short comparison rule across Choice, Answer and Review: apply the
  same evidence standard to related choices; distinguish actual use from a
  generic possible use, without treating specificity alone as evidence.

No new agent, schema, retrieval budget, mandatory evidence need, model, or
painting-specific answer/timestamp rule was added. Purpose inference and advisory
review comments remain allowed. The public app is unchanged. OpenAI Docs informed
the choice to make targeted prompt changes and evaluate them against the observed
failure; the diagnosis above comes from local traces and visual inspection.

## Validation

The offline focused suite passes 233 tests, including the existing downstream
evidence and review contracts. Fresh live results are recorded separately under
`artifacts/gpt54_v7_library_2026-09-08/`; the v6 run remains untouched.

### Fresh GPT-5.4 results

All four library questions completed with correct, matching reviewed answers.
These are exposed development examples, not held-out accuracy evidence.

| Example | Final choice | Cold total | QA portion | Estimated cost |
|---|---|---:|---:|---:|
| Sofa | Unwrap it | 40.55s | 26.64s | $0.12144 |
| Tree | Continue unwrapping presents | 32.07s | 19.15s | $0.09167 |
| Kitchen | C is cleaning dishes. | 51.23s | 28.20s | $0.15504 |
| Painting | To clean the paintbrush. | 60.48s | 34.11s | $0.16479 |

Total reported-usage-based cost: **$0.532938** for 62 HTTP attempts; no outstanding
or ambiguous reservations. Model, reasoning setting (`none`), media, options and
evidence budgets are unchanged. Fresh captioning means this is a full-system
comparison, not an ablation isolating which individual change caused improvement.

Painting now creates a `local_event` plan with its temporal dependencies intact,
without a planner repair. Retrieval selects scenes 7 and 8 (140–180 seconds),
rather than the three overview windows. The existing close-look pass identifies
brush/cup contact at approximately 179.75 seconds. The preliminary choice, Answer,
and both reviews select cleaning. The conclusion explicitly labels cleaning as
inferred and retains the missing immediate post-cup action as a limitation.

### Remaining imperfections

This is an improvement, not proof of perfect retrieval or review. The top cup-use
search still ranks a palette-work window above the actual cup window; the latter
arrives through the temporal neighbor search. The close-look's newly sampled
frames are earlier than the eventual cup contact, which is recognized in its
retained endpoint frame. The initial review also unnecessarily requests a direct
frame citation already reachable through an observation reference, causing one
explanation-only revision. Both reviews call the generic water-source alternative
unsupported because cleaning fits better, despite the instruction that a losing
choice may still have support. These issues do not change this run's selected
answer, but the reviewer is not an independent correctness guarantee.

The four-example library check passes; no broader held-out release claim is made
and the public Space remains unchanged.

Records: [summary](../artifacts/gpt54_v7_library_2026-09-08/summary.json),
[painting bundle](../artifacts/gpt54_v7_library_2026-09-08/cases/egoschema-painting/bundle.json),
and [configuration](../artifacts/gpt54_v7_library_2026-09-08/run-config.json).

Independent post-run audit reconciled all 62 requests, exact requested/reported
GPT-5.4 snapshot IDs, zero reported reasoning tokens, final Answer/Verifier pairs,
frame hashes, usage and ledger charges. All 66 frozen source/helper hashes and
89 recorded historical/annotation controls remained unchanged. Independent visual
inspection confirmed the cited 179.75-second image shows brush/cup contact, not
palette contact. It does not show the post-cup result, as the answer acknowledges.
