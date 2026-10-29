# Request-admission regression check

The completed 40-case run is now development-exposed. Its scores, raw outputs,
source snapshots and ledgers remain unchanged. This check is not a new accuracy
result and does not authorize another 40-case run or public deployment.

`scripts/replay_rejected_packs.py` reads the 15 audited local request rejections.
It reconstructs the old bound using only the captured pure schema converter and
system prompt, then serializes the exact same prompts, schemas and image files
through the current local adapter. It constructs no client and reads no API key.

The candidate changes the conservative structured-request admission allowance,
not the captured evidence or model. The report compares old/new admission and
bounds, prompt and schema hashes, image order and encoded image hashes. It also
checks that required fields, enums, bounds and closed-object validation survive.
Keeping required fields and `additionalProperties: false` preserves the relevant
[OpenAI Structured Outputs requirements](https://developers.openai.com/api/docs/guides/structured-outputs).
Admission success is not evidence of answer accuracy or provider acceptance.

Run the read-only replay with:

```bash
.venv/bin/python scripts/replay_rejected_packs.py
```

Only after the candidate is ready, `--write` saves a new exclusive report under
`artifacts/payload_replay_2026-09-08`; it cannot overwrite a previous report.

`--prepare-smoke` fixes exactly four existing library questions plus the five
terminal-contract failures from the completed evaluation. The separate proposed
smoke root is `artifacts/packing_contract_smoke_2026-09-08`. It uses pinned
`gpt-4.1-mini-2025-04-14`, two workers, at most nine once-only cold-index sessions,
40 HTTP attempts per session, 360 total attempts, and a $0.50 conservative cap.
Labels remain scoring-only. The separate `evaluate_request_contract_smoke.py`
runner requires explicit `--run` authorization; its default is read-only preflight.
It reserves $0.03 before every actual request in a new isolated accounting module,
covering the larger 65,536 structured allowance while preserving the $0.50 cap.
The execution gate requires nine normally completed matching answer/review pairs;
valid abstentions, nonabstaining answers, correctness, and library correctness are
reported separately. Passing this smoke never authorizes publication.
These are regression cases,
not a held-out cohort, and their results must never be merged with earlier scores.
