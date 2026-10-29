# Public structured-v7 / GPT-5.4 deployment

The user explicitly requested publication after reviewing the local/public
comparison and its remaining limitations. This supersedes the earlier instruction
to keep the public baseline unchanged. It is an owner-approved demo release, not
a claim that the historical held-out accuracy/completion gates have been met.

Target: `lqh52/videomasys-demo`.

## Before deployment / rollback reference

- Source and running revision: `3cfdfc4303b6de6a3516990793fdd0befd62cc2a`.
- Public visibility; CPU Basic; one replica; sleep setting 172800 seconds.
- Variables: `VLM_BACKEND=openai`, `VLM_MODEL=gpt-4.1-mini-2025-04-14`,
  `VLM_PROMPT_PROFILE=clean`.
- No `DEMO_MAX_*` overrides: existing defaults are five runs, 120 HTTP attempts,
  32 attempts per run. These are shared ledger allowances, not a dollar cap.
  The ephemeral ledger may reset when the container filesystem is replaced.

Rollback must restore both the prior source content and the above model/profile
variables. No secret values were read, copied, or recorded. Preserve hardware,
replicas, visibility and all other settings.

## Release inputs and checks

Publish the existing allowlisted application sources and reviewed library assets,
with `gpt-5.4-2026-03-05` and `structured-evidence-v7`. Do not include `.env`, local
artifacts, benchmark labels, tests or evaluation scripts. The four library entries
contain questions/options, not reference answers. UI styling and library assets
are unchanged.

The local v7 run completed all four exposed library questions correctly for
$0.532938. This is development evidence, not a new held-out benchmark. Existing
deployment/runtime tests pass (118 focused tests); the previous v7 pipeline suite
passes 233 tests. Official OpenAI documentation confirms the pinned snapshot,
image input, strict structured outputs and reasoning-disabled configuration.

The public runtime does **not** include the local evaluation's $2 budget ledger.
Existing request/run limits remain unchanged; no paid GPU or billing-setting
change is authorized by this deployment.

The canonical app origin currently fails normal TLS validation from this workspace.
No TLS bypass, alternate routing or DNS override is used. Hub management metadata
and buffered build/run logs are accessible. A hosted inference smoke test cannot
be claimed unless normal app-origin access succeeds.

## Published revision

Source commit: `6fd6f3d3f0f7f523a5b7beb1023a1714594b04b0`.
The upload reused the existing deployment allowlist, sent only 16 changed files,
and used the previous revision as a compare-and-swap parent guard. Application
sources matched the frozen v7 evaluation before upload. Existing library media,
thumbnails, UI style and dependency files were unchanged.

After the upload, only `VLM_MODEL=gpt-5.4-2026-03-05` and
`VLM_PROMPT_PROFILE=structured` were changed. A complete variable comparison
confirmed all other values stayed identical. No secret, hardware, sleep, replica,
visibility, billing or quota setting was changed.

## Post-deployment verification

- Hub source revision and running-container revision both equal the published
  commit; stage is `RUNNING`, visibility is public, CPU Basic remains selected,
  sleep is 172800 seconds, and the replica count remains one.
- All 66 allowlisted deployed files match local content hashes, including all
  eight library files. There are no mismatches.
- Deployment variables confirm OpenAI / full pinned GPT-5.4 / structured.
- Buffered build logs show completion with no traceback. Fresh run logs show
  Gradio's local-URL startup-ready message with no traceback.
- The final normal HTTPS check of the canonical `/config` endpoint still fails
  certificate validation in this workspace. Hosted UI and inference were therefore
  **not retested**; zero hosted analysis submissions were made by this deployment.
  No TLS bypass or local inference substitute was used.

Public links: [Space](https://huggingface.co/spaces/lqh52/videomasys-demo),
[direct app](https://lqh52-videomasys-demo.hf.space).
