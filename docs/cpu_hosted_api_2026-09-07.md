# CPU-hosted demo and pay-per-use visual inference

## Why this changes availability

The existing public Space reported `Scheduling failure: not enough hardware
capacity`, with no allocated hardware and `l4x1` requested. That failure happens
before app startup. The new default entry point starts Gradio on CPU Basic;
inference-provider availability affects analysis, not the page's GPU allocation.
CPU Basic can still sleep and Hugging Face can still experience service outages.
This is not an uptime guarantee.

The public app uses one explicit model route:
`Qwen/Qwen3-VL-8B-Instruct:featherless-ai`, through the Hugging Face router. No
larger-model or alternate-provider fallback is automatic. Provider-managed
weights are not represented as our old pinned local model revision.

## What stays and what changes

- Same Gradio layout, library thumbnails, planner, typed tool agent, retrieval,
  fresh evidence inspection, answer/review stages, and real prompt/output traces.
- Video extraction, SQLite cache, exact-cosine retrieval and orchestration run on CPU.
- MiniLM uses FastEmbed 0.8.0 / CPU ONNX rather than importing PyTorch. Its cache
  identity differs from the prior encoder implementation; mixed vectors are rejected
  through configuration signatures. Initialization is lazy.
- Only selected resized frames and agent text are sent to the VLM. Images use
  JPEG quality 90, at most 640×480 (512×384 for calls with more than six images).
  Per-call limit is 18 images; the full video is never submitted to the API.
- A shared HTTP client is reused within each analysis. Requests use greedy decoding
  with repetition penalty 1.08, matching the local anti-repetition setting.
- Raw traces include actual provider/model identifiers, HTTP attempts, latency,
  image counts and provider-reported usage. Usage is not an exact invoice estimate:
  the visual preflight reported only 43 prompt tokens despite receiving 18 images.

## Boundaries and cost controls

The initial shared allowance is five analysis starts and 120 HTTP attempts per
SQLite ledger, with at most 32 attempts per analysis. Failed/ambiguous attempts
are reserved before submission and are not refunded. These values are owner-side
environment settings, never browser inputs. There is no automatic daily refill.

The ledger survives ordinary process restarts while its file survives. CPU Basic
storage is ephemeral, so container replacement may reset it. This is therefore
**not a durable account-wide or dollar spending cap**. It does not control usage
outside this application. No auto-recharge or additional subscription is enabled.

Each model call has a 60-second caller deadline. At most one short retry is allowed
for connection failures, 429, 502, 503 or 504, or a fully received empty assistant
completion explicitly finished with `stop` and without an error, refusal, reasoning
or tool calls. All of these share the same two-attempt ceiling; a recovery request
can incur an additional charge. Long Retry-After values are respected
by stopping rather than retrying early. Ambiguous read/write/deadline timeouts are
not retried. Synchronous HTTP cancellation cannot guarantee the provider stops or
does not bill an already submitted request. Structural model repairs remain bounded
by the existing pipeline and count against the same allowance.

Rejected HTTP 200 responses retain safe shape/finish-reason/usage diagnostics,
without logging their bodies. See the [response-failure audit](hosted_response_recovery_2026-09-07.md)
for the incident's limits and verification evidence.

Credentials go only to the fixed HTTPS router without redirects. Error messages
never echo upstream request bodies, headers or tokens. A fine-grained inference-only
`HF_INFERENCE_TOKEN` Space secret is required; the broad local deployment token is
not copied into the public application. Without the secret, the UI and library load
and live analysis reports that configuration is missing.

## Verification evidence

Artifacts are in `artifacts/evaluation/hosted_api_2026-09-07/` and
`artifacts/hosted_cpu_ui_2026-09-07/`. These are development-library smoke tests,
not new holdout evaluation or a general accuracy benchmark.

- Initial real text preflight: HTTP 200, valid JSON, 15.31 seconds.
- Initial real 18-frame preflight: HTTP 200, valid JSON describing the gift video,
  3.64 seconds. This tests image transport, not full QA latency.
- CPU MiniLM cold setup/download plus three embeddings: 9.3 seconds locally;
  normalized 384-dimensional vectors and sensible semantic separation.
- `smoke_v1`: actual captioning, planner, compiler and retrieval succeeded, but
  scene-3 inspection repeated text and exhausted its output limit twice. It stopped
  without an answer. The initial API adapter had omitted the local repetition
  penalty; the retained failed artifact explains the subsequent fix.
- `smoke_v2_rp108`: two fresh live runs with the restored repetition penalty and new
  cache identity. Sofa multiple choice: "Unwrap it", 60.01 seconds, 13 model calls,
  selected scenes 1/2/3. Kitchen open-ended: washing kitchenware, 72.70 seconds,
  selected scenes 0/4/8 across the video thirds. Both completed answer/review stages
  and were accepted by the model reviewer. Both include new video indexing; these
  are local CPU clients calling the real hosted API, not measurements from the Space.
  Model review is not independent factual validation of every explanation detail.
- Desktop/mobile browser check: library selection, visible missing-token error,
  Run button recovery, and no horizontal overflow at 390px. Screenshots retained.
- Unit coverage includes denial before HTTP, retry accounting, ledger reopening and
  concurrent reservations, no GPU imports at startup, safe credential/error handling,
  session advancement across Gradio worker threads, cancellation cleanup and the
  unchanged public-deployment/hardware guards.
- Full repository test run: **577 passed**; relevant Ruff checks clean.

## Deployment state

Uploaded public Space commit `0bfe72751dbf981e6f18a7e6c24dcb019e53be32` and explicitly
requested `cpu-basic`. The first post-deployment API read confirmed `BUILDING`,
no current hardware, and requested hardware `cpu-basic` (not L4). Hugging Face set
the free CPU inactivity timeout to 172800 seconds. No inference secret was present
at that check, and no credentials were installed by this migration.
The build subsequently finished successfully. The final checked runtime was
`APP_STARTING`, still requesting `cpu-basic`, with no allocation/error reported
and no Space secrets. Anonymous page readiness and a live run from the deployed
Space are **not yet verified**. An early anonymous `/config` request timed out
during startup. Local real-provider QA tests above are not a substitute for that
deployment check. The temporary local UI server was stopped after testing.

### Follow-up: public deployment verified after the owner added the secret

On September 7, 2026, the owner added `HF_INFERENCE_TOKEN`; verification read only
the secret's name, never its value. The first runtime read briefly showed
`RUNNING_BUILDING` with old L4 current / CPU Basic requested. Before any intervention
or inference, a subsequent read confirmed `RUNNING` with both current and requested
hardware `cpu-basic`. No pause, restart, hardware change, credential write, or new
deployment was needed in this follow-up.

The public `/config` returned HTTP 200, Gradio 6.26.0, and 33 components. One real
sofa-example analysis was then submitted using a fresh **anonymous** browser inside
the Hugging Face wrapper, without local HF authentication. It completed in
**59.8 seconds**, selected **"Unwrap it"**, and displayed **Model check passed**.
Three expand/collapse cycles passed geometry checks, the footer was reachable via
the outer page scrollbar, and there were no browser errors. This consumed one
analysis from the shared application allowance; no second run was submitted.

Report and screenshot: `artifacts/cpu_space_2026-09-07/anonymous_demo/`.
Space commit remained `0bfe72751dbf981e6f18a7e6c24dcb019e53be32`. Public app readiness
and the secret-backed hosted inference path are now verified, superseding the
pending startup status above. Model-review agreement is not a general accuracy
guarantee, and the ephemeral application allowance is still not a billing cap.

## Operations

Deploy only after reviewing the allowlist:

```bash
.venv/bin/python scripts/deploy_zerogpu.py lqh52/videomasys-demo \
  --include-library --allow-public --hardware cpu-basic
```

The explicit hardware flag is necessary: ordinary code-only updates deliberately
preserve existing hardware. Never infer migration from README suggested hardware.
Confirm both requested and actual CPU hardware and an anonymous successful page
load. Add the inference-only secret in Space settings; do not paste it in chat or
commit it. Increasing allowance values is an owner action after reviewing billing.

Sources checked September 7, 2026:
[HF routed inference](https://huggingface.co/docs/inference-providers/en/providers/featherless-ai),
[HF billing](https://huggingface.co/docs/inference-providers/en/pricing),
[Featherless generation parameters](https://featherless.ai/docs/completions),
[Space hardware and billing](https://huggingface.co/docs/hub/spaces-gpus),
[FastEmbed](https://qdrant.github.io/fastembed/).
