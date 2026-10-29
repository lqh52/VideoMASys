# Small retrieval hybrid — local only

The `svp` opt-in profile now uses `svp-roles-v3`. This is a small retrieval/prompt
revision, not a new architecture. The default `clean` profile and public Space
are unchanged. The earlier [SVP experiment](svp_logic_2026-09-08.md) remains the
record of v1/v2 and their failures; its old evaluation batch is closed.

## What stays

Planner → tool agent → cached retrieval → visual inspection → answer → review.
The same pinned `gpt-4.1-mini-2025-04-14`, caption/vector cache, typed v3 program
schema, and maximum three selected windows / 60 seconds remain. No new agent,
action-memory database, full-video fallback, or observation loop was added.

## Small changes

- Keep SVP's ordered tools and optional source-scoped `locate`. A direct
  `locate(video)` followed by `inspect` is also a complete valid program.
- Use the previous retriever's target-aware temporal ranking: score eligible
  windows against the target interaction before applying the relation limit.
  The program's explicit time horizon is still binding.
- Select by evidence need, deduplicating candidates. Repeating a scene in
  several tool outputs must not give it extra votes that displace later evidence.
- Preserve representative overview retrieval for whole-video questions; do not
  manufacture before/after dependencies between general supporting activities.
- Teach precise anchors, optional scoping, and target searches with a small
  selection of fictional paired plan/program examples. Caption suggestions stay
  search hypotheses, not observations.

The prompt work follows [official OpenAI documentation on explicit instructions,
tool examples, and empirical evaluation](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-4.1).
The existing typed JSON program interface is retained; this is not an API migration.

## Evaluation

The [bounded protocol](svp_hybrid_evaluation_protocol_2026-09-08.md) permits two
new local runs, at most 24 actual HTTP attempts in total and 12 per run. Both use
identical historical caption/vector seeds. Previous clean and SVP-v2 outputs are
historical controls, not newly repeated paired runs or unseen holdouts.

### Actual results

All scene IDs below are zero-based trace IDs. All six rows used the same pinned
model and identical fixed cache inputs per question. The first four rows are
preserved historical results; only the last two are new runs.

| Case / version | Selected scenes | Selected duration | API attempts | Elapsed | Estimated USD |
| --- | --- | ---: | ---: | ---: | ---: |
| Gift / clean-v2 (historical) | 1, 2, 3 | 54.01s | 9 | 39.06s | $0.01607 |
| Gift / SVP-v2 (historical) | 0, 1, 2 | 54.01s | 9 | 32.57s | $0.01500 |
| Kitchen / clean-v2 (historical) | 2, 5, 6 | 60s | 8 | 35.40s | $0.01167 |
| Kitchen / SVP-v2 (historical) | 2, 4, 8 | 60s | 9 | 31.70s | $0.01170 |
| Gift / hybrid-v3 | 0, 1, 3 | 54.01s | 9 | 39.20s | $0.01676 |
| Kitchen / hybrid-v3 | 2, 4, 8 | 60s | 8 | 29.95s | $0.01241 |

Both new runs produced the correct main conclusion, with no schema, execution,
answer-revision, or transport retries. Each retained the existing single visual
refinement call. Total: **17 HTTP attempts, approximately $0.02917**. The two
admitted sessions close the new batch; seven unused request slots do not authorize
another run. No model reasoning tokens were reported.

Costs use reported token counts and [OpenAI's GPT-4.1 mini rates](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
of $0.40 input / $0.10 cached input / $1.60 output per million tokens, checked
2026-09-08. These are estimates, not account invoices. New gift usage was 37,572
input and 1,080 output tokens; kitchen 27,155 input and 968 output; neither had
reported cached input. Historical v2 runs did have cached input. Different cache
hits and service conditions mean this is not a controlled latency/cost benchmark.

### Gift: missing outcome recovered, early setup still retained

The actual generated program was:

```python
present_pickup_sofa = locate(
    source="video", query="boy picks up present and moves to sofa",
    limit=1, evidence_need_ids=["present_pickup"]
)
after_pickup = relate(
    source="present_pickup_sofa", relation="after", horizon_seconds=60.0,
    limit=4, evidence_need_ids=[]
)
sofa_present_interaction = locate(
    source="after_pickup", query="boy interacts with present at sofa",
    limit=2, evidence_need_ids=["sofa_interaction"]
)
inspect(
    sources=["present_pickup_sofa", "sofa_present_interaction"],
    evidence_need_ids=["present_pickup", "sofa_interaction"]
)
```

The target query ranks scene 3 first (cosine 0.62254), followed by 1 (0.50192),
0 (0.48289), 2 (0.44322), and 4. The relation's four-result limit is applied after
that ranking. The final scoped locate retains 3 and 1, and the explicit anchor
retains 0. The resulting answer selects **“Unwrap it.”**

Visual review of [the actual selected frames](../artifacts/svp_hybrid_2026-09-08/svp__svp-roles-v3__sofa_mc/review-frames.jpg)
confirms that the recovered 54.01–72.02s window includes separated wrapping material
around 57.77/64.77s and the contents afterward. This is a concrete improvement over
the prior SVP candidate, which excluded that window from inspection.

It is not a clean win over the previous clean retriever: the compiler still
combines pickup and arrival in its query despite the arrival-focused instruction,
so scene 0 remains selected instead of clean's scene 2. The new final answer also
labels missing alternatives as “contradicted” based on absence, and its explanation
for the lady option is not established by the sampled images. Some exact inspection
action/timestamp claims remain too strong. The verifier accepted these issues.
Correct choice and recovered evidence therefore do not imply a full grounding pass.

A read-only replay of the exact old SVP-v2 program under the new retriever still
selects 0/1/2. Its 30s horizon excludes scene 3 (36.01s gap after anchor 0), and
the runtime does not secretly widen it. The new live result changes both the
generated program and selection policy; it is not an isolated causal measurement
of the ranking change alone.

### Kitchen: straightforward global retrieval, same useful evidence

The new plan has two independent needs: main activities and object interactions.
It does not invent a temporal dependency between them. Its actual program is:

```python
activity_overview = overview(
    source="video", limit=6, evidence_need_ids=["main_activities"]
)
object_interactions = locate(
    source="activity_overview", query="person interacting with kitchen dishes and utensils",
    limit=3, evidence_need_ids=["object_interactions"]
)
inspect(
    sources=["activity_overview", "object_interactions"],
    evidence_need_ids=["main_activities", "object_interactions"]
)
```

The selected 40–60s, 80–100s, and 160–180s windows support the main answer:
cleaning dishes and kitchenware. This preserves the SVP-v2 representative scenes
while removing its unnecessary temporal relation and compiler repair.

[Visual review](../artifacts/svp_hybrid_2026-09-08/svp__svp-roles-v3__kitchen_open/review-frames.jpg)
shows container scrubbing, plate/cup handling and rinsing, and glass scrubbing.
The inspector still claims a brown plate is scrubbed at 84.15s when the sample
shows it being held near the rack, and claims pan/pot placement at 175.85s that
the sampled images do not clearly establish. The final answer avoids the pan
claim but still calls items “cleaned” without demonstrating cleaning completion.
Again, the model's accepted review is not a full grounding pass.

## Validation and recommendation

**998 offline tests passed**; Ruff passed for changed Python files. Added tests
cover target ranking before temporal limits, unrelated-query isolation, explicit
horizons and empty scopes, duplicate-source invariance, feasible need/temporal
coverage, global representative selection, and the controller's treatment of
unselected redundant sources. The existing repair limit did not increase.

The runtime now retries for missing required needs, temporal branches, or global
coverage—not merely because a redundant broad source supplied no final window.
That unused-source information remains visible in the real trace.

Source snapshots/hashes were frozen across both runs and still match the local
implementation. The old ledger hash is unchanged at 6 sessions / 54 requests;
the new ledger records 2 sessions / 17 requests. The public Space was not touched.

Keep this small hybrid as the local candidate: it recovers the gift outcome and
produces a more appropriate global program without adding stages. Do not call it
a general accuracy upgrade or promote it automatically. The remaining problems
are precise visual descriptions, uncertainty calibration, and avoiding redundant
setup evidence—not a reason to add another architectural layer.

Artifacts: [gift report](../artifacts/svp_hybrid_2026-09-08/svp__svp-roles-v3__sofa_mc/report.json),
[kitchen report](../artifacts/svp_hybrid_2026-09-08/svp__svp-roles-v3__kitchen_open/report.json).
Each directory includes the complete bundle, raw trace, exact source snapshots,
immutable inputs, and sampled frames. Local opt-in remains
`VLM_PROMPT_PROFILE=svp`; `clean` remains the default.
