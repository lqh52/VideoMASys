# Clean role prompts and a stronger non-reasoning model

## Change and scope

The active clean prompts are centralized in
[`clean_prompts.py`](../src/videomasys/zerogpu/clean_prompts.py), profile `clean`,
version `clean-roles-v2`. Each role has its job, supplied context, and essential
evidence/output rules. Worked examples and repeated case-specific instructions
are absent. The previous role text and paired synthetic examples remain available
with the owner setting `VLM_PROMPT_PROFILE=legacy`. The public Space was temporarily
restored to that previously tested profile during the network block below. Local
key-based tests subsequently completed; the final selection is clean v2 / GPT-4.1
mini. The historical failures below remain part of the record.

The schemas, typed `locate` / `relate` / `inspect` executor, embedding model,
sampling/refinement, evidence limits, exact-choice selection, verifier and one
correction ceiling are unchanged. Clean captions have a separate versioned cache
identity. Actual prompts, response schemas, outputs and profile/version remain in
the trace. The historical ICL-only evaluation script explicitly selects `legacy`.

This follows OpenAI's recommendation to start with focused output contracts and
evaluate small prompts, rather than adding reasoning instructions or broad agent
scaffolding. [GPT-5.4 prompting guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4),
[GPT-4.1 guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-4.1).

## Model and safety settings

Comparison candidates:

- `gpt-4.1-mini-2025-04-14`: non-reasoning model, `temperature=0`; no unsupported
  reasoning parameter. Standard input/cached-input/output rates: $0.40/$0.10/$1.60
  per million tokens. [Model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
- `gpt-5.4-mini-2026-03-17`: pinned snapshot, explicit `reasoning_effort=none`,
  `temperature=0`. Rates: $0.75/$0.075/$4.50 per million tokens.
  [Model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

Both use the existing fixed OpenAI Chat Completions endpoint, strict schemas for
structured stages, standard service tier, `store=false`, bounded output/image
sizes and retries. No raw inference endpoint or browser key exposure was added.
The selected model never silently falls back to another model or provider.

At most four new analysis submissions are planned: two questions on each model.
No failed analysis is automatically resubmitted. The existing shared 5-run,
120-request, 32-request-per-run limits are unchanged; ordinary restarts can replace
the ephemeral ledger, so the evaluation budget is also tracked across restarts.
CPU Basic hardware, visibility, sleep controls and account billing are unchanged.

## Evaluation method

Use the ordinary public Gradio Run analysis workflow, with the existing opt-in
browser smoke helper. Preserve raw prompts/responses and failed results. No answer
labels enter inference. Test the same questions and choices:

1. NExT-QA gift/sofa, library index 0, unchanged question and choices.
2. EgoSchema kitchen, index 2, no choices: “What is the person mainly trying to
   accomplish across the video? Describe the visible actions that support your answer.”

Historical GPT-4.1 mini runs are retained at
`artifacts/openai_2026-09-07/mini_sofa_mc` and `mini_kitchen_open`.
New runs go in `artifacts/clean_prompts_2026-09-07/`.

Grade primary answer, visible support/temporal coverage, and model review separately.
Verifier acceptance is not a correctness label. These are end-to-end comparisons:
prompt/model changes may change cached captions, retrieval and refinement, so the
models need not receive identical frames. Two cases do not establish benchmark
accuracy, statistically reliable latency, or parity with Qwen3-VL-30B.

The offline helper [`summarize_prompt_evaluation.py`](../scripts/summarize_prompt_evaluation.py)
extracts timings, actual token usage, prompts, selected windows, repairs and all
answer/review revisions without network calls or mutating the source artifacts.
Costs are estimates from returned token usage and standard rates, not invoices.

## Offline verification

800 tests passed before live evaluation; Ruff passed. Representative question/plan
fixtures reduce planner prompt characters by about 82% and compiler characters by
66–67%. This is not a claim of equivalent token, cost or latency reduction.

The first code upload was `0a0f84ee6d3b0ab74d470163719adb3bbafb6455`; no analysis
was submitted on it. Review restored explicit planner `max_intervals=3` and
`max_inspected_seconds=60` before live evaluation. Evaluation source commit:
`6a59c5cdf461708e3f4f848c41ac498ae9888600`.

## Initial live results (clean v1)

Two new analyses were submitted, both on GPT-4.1 mini with `clean-roles-v1`.
All 25 HTTP attempts returned responses; there were no transport retries or
reasoning tokens. Total estimated model cost was **$0.023906**.

| Question/configuration | Result | Total time | API attempts | Estimated USD |
|---|---|---:|---:|---:|
| Gift, historical Mini/legacy | Correct choice; grounding errors | 40.8s | 14 | $0.0216512 |
| Gift, Mini/clean v1 | Compiler validation failure; no answer | Not recorded | 8 | $0.006868 |
| Kitchen, historical Mini/legacy | Correct main activity | 35.7s | 16 | $0.0162264 |
| Kitchen, Mini/clean v1 | Correct main activity; grounding errors | 46.9s | 17 | $0.017038 |
| GPT-5.4 mini/clean v1 | Browser connection failed **before inference** | — | 0 | $0 |

### Gift failure and narrow follow-up

The planner correctly described an anchor before its outcome. The tool agent
misread the direction from the source: it generated `relate(before)` to retrieve
the outcome from the anchor, rather than `relate(after)`. Its one semantic repair
repeated the same program despite the validation message. The controller stopped;
no retrieval or answer was fabricated. The complete failure is retained in
`artifacts/clean_prompts_2026-09-07/mini_sofa_mc`.

The browser helper then waited for an answer panel that correctly did not exist.
That secondary timeout did not cause the pipeline failure. The helper now captures
an absent answer as null, preserving the original failure without that wait.

Clean v2 adds only this generic clarification to the compiler instructions:
`relation` places the target relative to the source; if A precedes B, retrieving B
from A uses `after`, not `before`. At this point it was a proposed correction with
offline coverage, not yet live-validated. It changes the prompt version/cache
signature. The subsequent v2 gift runs both produced valid programs without repair.

### Kitchen visual audit

The clean answer correctly identifies washing kitchenware. It still claims a
brush scrubs the red board, while the selected image around 128.05s shows a blue
sponge. The inspector also invents placing a green board on a red board; that
claim is not repeated in the final answer. The same-model verifier accepts the
incorrect brush claim. Independent inspection of all 18 final images and the
18 baseline images found no grounding improvement.

Selected windows changed from 0/3/8 (0–20, 60–80, 160–180s) to 1/5/6 (20–40,
100–120, 120–140s). Both satisfy the existing three-thirds rule, but the clean run
does not include the final 40 seconds. The clean run adds a refinement call. Prompt
characters across the run fall from 32,011 to 20,310, yet total tokens/cost and
latency rise. Image selection, refinement and serial calls still matter; text
shortening is not sufficient evidence of an efficiency gain.

### Stronger-model test blocked

After setting `VLM_MODEL=gpt-5.4-mini-2026-03-17`, the public browser test failed
before loading the model badge or pressing Run analysis (`inference_invoked=false`).
The artifact is `artifacts/clean_prompts_2026-09-07/stronger_sofa_mc`.

Read-only checks found the app domain resolving to `146.112.61.110` and presenting
a certificate issued by `Cisco Umbrella Secondary SubCA syd-SG`, not a trusted
Hugging Face connection from this workspace. Hugging Face's management API still
reports the CPU Space running. TLS verification was not disabled, DNS was not
overridden and no network filter bypass was attempted. The OpenAI key exists only
inside the Space; it was not extracted. No inference was submitted to GPT-5.4 mini,
so its API compatibility, performance and quality are not yet live-verified here.

### Public configuration and next test at the network block

The public variables were restored to `VLM_BACKEND=openai`,
`VLM_MODEL=gpt-4.1-mini-2025-04-14`, `VLM_PROMPT_PROFILE=legacy`.
This retains the previously tested behavior, not a claim that its answers are
perfect. The source contains the clean candidate and stronger adapter for a
follow-up comparison. No hardware, secrets, payment controls or usage limits were
changed.

To finish, the app domain needs legitimate network access from this workspace,
or the owner can supply the OpenAI key through the local environment (never chat)
for a bounded local comparison. Re-test clean v2 on the failed gift case and test
the same gift/kitchen questions on the stronger non-reasoning model. Do not infer
that either cleaner prompts or a larger model improved accuracy without those runs.

## Resumed local evaluation: clean v2

The owner supplied `OPENAI_API_KEY` in the workspace `.env`, enabling direct OpenAI
tests without accessing the network-filtered Space app domain. The file was made
owner-readable/writable only (0600); the runner reads just that key with dotenv
interpolation disabled, never loads other file settings, and does not log, hash or
upload the key. Existing deployment allowlists exclude `.env` and evaluation data.

[`evaluate_openai_prompts.py`](../scripts/evaluate_openai_prompts.py) runs exactly
one explicit model/case per invocation. All four invocations share the same local
SQLite ledger, capped at four runs, 120 attempts, 32 attempts per run. Existing
model/case directories cannot be overwritten or automatically rerun. This separate
evaluation allowance does not modify the public Space's limits or account billing.
The earlier failed v1 and blocked browser attempts remain in their original paths.

All four analyses completed, each with its correct primary answer and an accepted
same-model review. **Independent visual audits still found grounding errors.**
No reasoning tokens or transport retries were reported. API-reported snapshots
matched the pinned models; GPT-5.4 mini accepted `reasoning_effort=none` and
`temperature=0` in actual requests.

| Model / case | Time | Calls | Semantic repairs | Selected windows | Estimated USD |
|---|---:|---:|---:|---|---:|
| GPT-4.1 mini / gift MC | 42.006s | 14 | 0 | 1 / 2 / 3 | $0.017838 |
| GPT-5.4 mini / gift MC | 38.451s | 14 | 0 | 2 / 3 / 4 | $0.03355725 |
| GPT-4.1 mini / kitchen open | 47.571s | 17 | 0 | 2 / 5 / 6 | $0.0132416 |
| GPT-5.4 mini / kitchen open | 49.026s | 18 | 1 | 0 / 4 / 7 | $0.03307875 |

Total: **63 actual request attempts, estimated $0.0977156**. Including the two
earlier clean-v1 runs, this prompt experiment used 88 attempts and an estimated
$0.1216216. Historical legacy/Luna experiments are not included in those totals.
Cached-input discounts differed between models, so the observed roughly 2.1×
aggregate price ratio is not a general per-query price guarantee.

Full local artifacts are in `artifacts/local_clean_v2_2026-09-07/`, under
`<pinned-model>__sofa_mc` and `<pinned-model>__kitchen_open`. Each contains
`manifest.json`, `bundle.json`, `raw-trace.json` and `report.json`. Actual final
images are retained in the model-specific cache, with paths in the bundle/report.

### Grounding and reliability findings

- Gift: both models select “Unwrap it.” The scene-3 images are byte-identical
  across models and show paper separation and revealed fabric-like contents.
  GPT-4.1 retrieves arrival window 1; GPT-5.4 omits it. Refinement reduces Mini's
  arrival context to endpoints plus a close interaction. Both answers nevertheless
  make relocation claims without appropriately citing the anchor, and both misuse
  absence as contradiction. Mini even denies a woman interaction that is visible
  in its scene-1 frames. GPT-5.4's inspector invents carrying a wrapped gift back
  toward the tree. Neither verifier catches the material errors.
- Kitchen: both identify washing kitchenware. GPT-4.1 repeats the incorrect brush
  identification on the red board and adds a questionable “visibly cleaner” claim.
  GPT-5.4's final answer avoids those details and is better calibrated on this
  example. Its inspector still merges different objects/actions in scene 7:
  it claims sponge-on-tray contact and a tray remaining in the final frame, while
  the images show tray handling, an empty sink, then a different clear container.
  Reviewer acceptance does not validate those observations.
- The concise relation-direction clarification worked on the follow-up Mini gift
  run: the tool program was valid without repair. One follow-up is evidence of
  success on that case, not a universal fix.
- GPT-5.4's kitchen compiler initially produced an unnecessary fourth locate,
  interpreting “one locate per need” and the three whole-video probes as additive.
  Static validation rejected it; its one repair returned three valid locates.
  The original failure remains visible in the trace.

### Decision and reproducibility

Keep **GPT-4.1 mini + clean v2** as the public default: it satisfies the user's
clean-prompt preference, completed these two cases without semantic repair, and
cost less. Keep GPT-5.4 mini as an explicitly selectable, tested non-reasoning
alternative through the owner `VLM_MODEL` setting. Its better kitchen final wording
does not establish a consistent quality gain across the two tasks, and its latency
was mixed. No sampling, retrieval algorithm or evidence-budget change was used to
make a model appear better.

These local timings include video preparation, indexing, embedding initialization
and all model stages. They are not Gradio/browser latency measurements. Different
model captions/plans selected different frames; this is an end-to-end comparison,
not a same-frame model benchmark. Do not treat earlier Space timings or two
examples as a statistically controlled latency/accuracy result.

The first local manifest captured the runner before a one-line cleanup-error
classification fix; later runs captured it afterward. That fix only marks an
exception during cleanup as a failure. No pipeline source, prompt or model setting
changed between these four successful runs. Final regression verification:
**807 tests passed; Ruff clean.**

Example explicit invocation (billable; refuses an existing model/case directory):

```bash
.venv/bin/python scripts/evaluate_openai_prompts.py --run \
  --model gpt-5.4-mini-2026-03-17 --case sofa_mc \
  --output-root artifacts/local_clean_v2_2026-09-07
```

The four-run ledger is now exhausted. Do not reset it or change output roots to
bypass the completed evaluation's budget; any further experiment needs a newly
agreed scope. No further paid tests were submitted after this comparison.

Final Space deployment: `3cfdfc4303b6de6a3516990793fdd0befd62cc2a`. The management
API reports `RUNNING`, public, CPU Basic current/requested, sleep time 172800s,
`VLM_BACKEND=openai`, `VLM_MODEL=gpt-4.1-mini-2025-04-14`,
`VLM_PROMPT_PROFILE=clean`. Repository listing confirms neither `.env` nor local
evaluation artifacts were uploaded. The code/model path is live-tested locally;
a fresh embedded browser check remains unavailable from the network-filtered
workspace. No Gradio layout or scrolling code changed in this work.
