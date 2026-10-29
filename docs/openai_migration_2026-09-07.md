# Direct OpenAI demo backend — 7 September 2026

## Current usage inventory and target mapping

The owner requested a cheap OpenAI backend after invalid hosted responses, with
Qwen3-VL-30B performance as a target. Existing recorded Space evaluations used
Qwen3-VL-8B, **not 30B**. There is no controlled head-to-head result establishing
parity with 30B. Small example checks cannot establish benchmark equivalence.

| Usage | Previous CPU/API backend | Selected OpenAI backend |
| --- | --- | --- |
| Window captions and fresh frame inspection | Qwen3-VL-8B / Featherless | Pinned GPT-4.1 mini, image input |
| Planner and typed tool compiler | Same Qwen model, text prompts | Same Mini model, strict JSON |
| Choice selection | Qwen, exact choice text | Mini, exact choice text |
| Answer and review | Qwen, images plus text | Mini, images plus strict JSON |
| Semantic search | CPU MiniLM / FastEmbed | Unchanged; no OpenAI embedding charge |
| Video processing, cache, retrieval, tool execution | CPU / SQLite / deterministic controller | Unchanged |

The request is a compatibility migration, not a blind model-name substitution.
Luna was tested first as the cheapest starting point. The Space is now explicitly
configured with `gpt-4.1-mini-2025-04-14` after testing both on the same two questions.
Neither parameter count nor those few examples establishes equivalent video reasoning.
The adapter allows only these two low-cost models, selected by the owner-controlled
configuration; there is no automatic model/provider fallback. Historic Qwen evaluations,
local Qwen runtime, server
prototype, and their fixtures remain unchanged.

## Changes and compatibility checks

- Select `VLM_BACKEND=openai` explicitly. Use server-side `OPENAI_API_KEY` only for
  the fixed OpenAI endpoint. The original `api` route still reads only
  `HF_INFERENCE_TOKEN`. Adding a secret alone does not switch providers.
- Use Chat Completions to retain the existing bounded transport and stateless
  stage contract. These are not native API function-tool loops: the model emits
  validated tool data, and the application executes its typed retrieval functions.
  No generated code is executed, and no new API-native multi-agent beta is enabled.
- For Luna, explicit `reasoning_effort=none` preserves the previous non-reasoning
  latency class and small per-stage completion limits; Luna would otherwise default
  to medium. Mini is non-reasoning and omits that parameter, using `temperature=0`.
  Use `max_completion_tokens`, not the HF route's `max_tokens` or
  `repetition_penalty`. Standard service tier only; no priority, Pro, or escalation.
- Use `store=false`, stateless calls, and no server-side conversation or reasoning
  replay. Preserve the existing resized frames and make image detail explicit.
- Pass Pydantic schemas through an explicit structured-generation callback and
  strict API response format. Closed objects and required properties preserve
  nullable field types, references, and validation constraints. Keep Pydantic and
  business validation, including evidence IDs, dataflow and citation restrictions.
- Keep exact-choice generation as plain text. Strict generation removes the
  redundant full JSON schema from the prompt and records it separately in the
  trace. Existing paired synthetic planner/compiler examples stay unchanged;
  no benchmark answers are added to prompts and no broad prompt rewrite is made.
- Separate all three model/cache identities, so prior Qwen or Luna captions and
  indexes do not silently serve as Mini results. Luna's alias is provider-managed,
  not immutable. Mini uses an exact dated API snapshot; its serving implementation
  is still provider-managed. Record and validate the actual returned model ID.
- Preserve bounded images, prompt/context allowance, per-call deadlines, retries,
  single-analysis concurrency, and shared request admission. Each HTTP attempt
  reserves allowance before submission. No unbounded retry or answer substitution.

These choices follow the OpenAI Docs skill's compatibility guidance; strict
formatting improves syntactic reliability, not the truth of observations or answers.

## Cost and hosting controls

The selected Mini snapshot's standard rates are $0.40 / million input tokens,
$0.10 / million cached input tokens, and $1.60 / million output tokens.
The initial Luna comparator's official page lists $0.20 / million input tokens,
$0.02 / million cached input tokens, and $1.20 / million output tokens. Cache writes
cost 1.25× input. Image inputs also consume tokens. The app stays far below the
documented long-context threshold; no explicit cache feature is added in this
migration. Any measured cost calculated from returned usage is an estimate, not
an invoice or a guarantee about timeout/failed-request billing.

The Space stays on CPU Basic with its existing sleep setting and one replica.
No subscription, credits, paid hardware, auto-recharge, or account billing setting
is changed. OpenAI usage is billed to the key's OpenAI project/account; Hugging
Face credit is separate.

Application defaults remain 5 run starts / 120 HTTP attempts shared across users,
32 attempts per analysis, with the existing 600-second analysis admission deadline.
This is **not a dollar spending cap**. The default SQLite ledger lives in `/tmp`
and can reset when the container is replaced. No limits are raised or ledger
deleted to make an evaluation pass.

## Validation

Offline checks completed:

```text
.venv/bin/pytest -q
755 passed in 23.37s

.venv/bin/ruff check src tests scripts deploy/zerogpu/app.py
All checks passed!
```

This includes 96 Luna/OpenAI adapter/schema tests, 12 pinned Mini tests, and all
92 legacy HF adapter tests,
strict callback and semantic-repair regressions, credential isolation, admission
and cleanup, an actual adapter/MockTransport/UsageGuard integration, provider-aware
UI and CPU-only entrypoint checks, and local browser tests of evaluation overrides.
Fixtures are explicitly labeled; none of those checks establishes answer quality.

Deployed source revision: `0d2d9184ff7281f5238ad3bec434c365abbe9895` to the
public `lqh52/videomasys-demo` Space, then explicitly set `VLM_BACKEND=openai`.
Hardware remains CPU Basic, one replica, with the existing 172800-second sleep
setting. Both provider secret names were checked without retrieving or printing
their values. The initial live deployment was verified RUNNING at that source SHA.

### Initial Luna live evaluation

Two fresh anonymous-browser analyses were run through the Space, without
extracting its API key. Both used a new model-specific video index. All 32 HTTP
attempts returned 200, all structured stages validated on the first attempt, and
both browser scrolling/expand-collapse checks passed. No transport retries occurred.

| Case | End-to-end | Result | API calls | Estimated API cost |
| --- | --- | --- | --- | --- |
| NExT-QA sofa, multiple choice | 52.6s | Selected `Unwrap it`, but draft and review failed to recognize visible unpacking; left Needs review | 16 | $0.011784 |
| EgoSchema kitchen, open-ended | 40.1s | Washing dishes/kitchenware, supported with early/middle/late observations; model review passed | 16 | $0.007759 |

The source question for the open-ended case was replaced explicitly with
“What is the person mainly trying to accomplish across the video? Describe the
visible actions that support your answer.” All options were cleared; no hidden
multiple-choice answer was sent.

Local artifacts: `artifacts/openai_2026-09-07/{sofa_mc,kitchen_open}/` contain actual
raw traces, browser reports and screenshots. Sofa used scenes 2, 3, 4; kitchen
used scenes 0, 4, 8. Luna did not fix the sofa perception problem just by changing
providers. A read-only reconstruction of the actual sampled sofa frames shows
wrapping separation and a distinct garment, contradicting the draft's blanket
claim of no opening evidence. The retrieval also omitted the actual move-to-sofa
window, and focused refinement reduced earlier context. None of these findings
authorizes forcing a known benchmark answer or weakening the verifier.

Usage totals: sofa 50,338 input / 1,973 output, with 4,608 cached reads and 3,564
cache-write tokens; kitchen 29,246 input / 1,430 output, 0 cached reads and 3,870
cache-write tokens. Estimated combined cost is $0.019543, calculated from returned
usage at the fetched rates, including cache write premiums. It is not an invoice.
The shared ledger after these runs showed 2 run starts and 32 request attempts;
limits were unchanged and no ledger was cleared.

### Additional low-cost comparator

Because Luna missed positive visual evidence, the app now also allows an explicit
GPT-4.1 mini comparator, pinned to `gpt-4.1-mini-2025-04-14`. This is not automatic
fallback. Its non-reasoning request omits `reasoning_effort`, uses `temperature=0`,
and retains the same schemas, prompts, sampling, per-call cap and shared allowance.
Standard rates are $0.40 input / $0.10 cached input / $1.60 output per million tokens.
The comparator source was deployed at
`3d5cfdd12d385c8cf296fe7dadf2ee2b7879d339`, then `VLM_MODEL` was explicitly set to
`gpt-4.1-mini-2025-04-14`. That revision was verified RUNNING on public CPU Basic,
one replica, same sleep setting. This is the final active Space configuration.

Both Mini runs used the same questions, choices, prompts, sampling policy and
evidence limits as the Luna runs, with new Mini-specific indexes. They are
end-to-end comparisons, **not identical-frame controlled model evaluations**:
model captions and plans change retrieval and refinement choices.

| Case | End-to-end | Main answer / model review | Calls | Estimated API cost |
| --- | --- | --- | --- | --- |
| NExT-QA sofa, multiple choice | 40.8s | `Unwrap it`; model review passed, with grounding problems found in the audit below | 14 | $0.021651 |
| EgoSchema kitchen, open-ended | 35.7s | Dishwashing and cleaning kitchenware; model review passed | 16 | $0.016226 |

All 30 Mini requests returned 200; no JSON repairs or transport retries occurred.
Both embedded-page expansion/scrolling checks passed. Actual traces and browser
reports are in `artifacts/openai_2026-09-07/{mini_sofa_mc,mini_kitchen_open}/`.
Mini sofa used scenes 0, 3, 4; kitchen used 0, 3, 8. Sofa consumed 48,724 input and
1,351 output tokens; kitchen consumed 35,878 input and 1,172 output, with no cached
reads in either run. At the fetched rates these two runs cost an estimated
$0.037878 combined. All four live evaluations together cost approximately
$0.057420, not an invoice or durable future spending cap.

The updated container's ledger showed 2 runs / 30 requests after the Mini tests;
the earlier container's ledger showed 2 / 32 after Luna. This demonstrates the
documented ephemeral-ledger reset on normal deployment. No ledger was manually
deleted and no limits were increased. Three further run starts remained in the
current ledger at measurement time; public visitors may consume them.

### Remaining quality gaps and decision

Keep the pinned Mini configured as a low-cost working candidate, **not** as proof
of robust grounding or Qwen3-VL-30B parity. It got both main answers right in these
checks, but the independent read-only sofa audit found:

- Retrieval still omits the actual move-to-sofa window. Scene 3 provides genuine
  unpacking evidence, but the inspector incorrectly places arrival at the sofa
  there; the boy is already beside it in those frames.
- The scene-4 observation incorrectly describes departure from the Christmas
  tree. The last sample still shows him kneeling beside it. Refined scene 0 shows
  holding the present by the tree, not arrival at the sofa.
- The answer labels unseen girl/lady/train alternatives as contradicted, rather
  than merely unobserved. Its same-model verifier fails to catch these errors.

No prompts, labels, evidence, or answers were altered to force a pass. Correct
option selection and a passed model review are reported separately from factual
grounding. A larger, controlled hold-out evaluation against the actual 30B model
is still needed before claiming matching performance. Improving window coverage,
object-state recognition and verifier discrimination remains relevant even with
the more reliable API/schema transport.

## Official sources

- [GPT-5.6 Luna model, pricing and modalities](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [GPT-4.1 mini model, pricing and snapshot](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
- [GPT-5.6 migration guidance](https://developers.openai.com/api/docs/guides/model-guidance?model=gpt-5.6)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Image inputs and detail](https://developers.openai.com/api/docs/guides/images-vision)
- [Chat Completions request reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

Sources fetched on 7 September 2026. Prices and alias behavior may change.
