# Structured evidence evaluation protocol

**Closed:** both authorized sessions were used, consuming 16 actual HTTP attempts
(nine sofa, seven kitchen). Both v1 runs stopped at answer validation. No answer
was substituted and neither reached review. The remaining eight attempts are not
available for another run. The runner remains pinned to v1 and rejects the newer
v2 profile; corrected v2 has offline validation only. Historical artifacts and
source snapshots preserve the actual tested version.

This is a new, explicitly bounded local evaluation of `structured-evidence-v1`,
using the `structured` profile and pinned `gpt-4.1-mini-2025-04-14`. It does not
deploy or change the public Space. Only the known `sofa_mc` and `kitchen_open`
regression cases are authorized, once each.

The new ledger is `artifacts/structured_evidence_2026-09-08/usage.sqlite3`, capped
at **two sessions, 24 actual HTTP attempts total, and 12 attempts per session**.
Failed requests, transport retries, and schema repairs count. There are no refunds,
resets, fallback models, automatic reruns, or automatic batch extensions. Two used
sessions close the batch even if fewer than 24 attempts were consumed. These are
local request limits, not an OpenAI account billing cap.

The hybrid batch remains closed at **two sessions and 17 attempts**; the earlier
SVP batch remains closed at **six sessions and 54 attempts**. Both ledgers and all
historical control artifacts are read-only. Each candidate uses the same immutable
historical video, scene captions, and normalized cached vectors as its hybrid-v3
control. Historical answers are not included in model inputs.

The tested hybrid planner/tool prompts and deterministic retriever must retain
their exact historical hashes. The treatment is downstream: typed observations,
frame-linked claims, explicit option support, complete review coverage, and
refinement using the exact six final images shown to the subsequent answer and
review calls. Actual retrieved windows can still vary between model executions;
this is not a fixed-frame replay or a fresh paired control run.

The runner snapshots source files and freezes their hashes across both candidate
cases. Each case has an exclusive artifact directory containing the input seed,
source snapshots, manifest, raw trace, result bundle, and report. Real request,
image, and schema hashes and actual transport usage are retained through the
existing guarded inference adapter. Execution completion and model review
acceptance are reported separately from answer correctness.

Run `scripts/evaluate_structured_evidence.py --case sofa_mc` for read-only preflight;
it does not read owner credentials, load embeddings, create a ledger, or call a
model. The owner must explicitly add `--run` to admit one paid local evaluation.
