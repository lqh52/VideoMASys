# Small hybrid: bounded local evaluation

The user's new **“Try that”** request authorizes this separate, small comparison.
It does not reopen the previous SVP batch, which remains closed at **6 sessions
and 54 HTTP attempts**. That ledger and all prior artifacts are read-only inputs.
The public Space, its deployment, settings, and usage allowances stay unchanged.

**Completed and closed:** the two runs consumed 17 HTTP attempts (9 gift, 8
kitchen). The remaining seven request slots do not permit another session. See
the [results and visual audit](svp_hybrid_2026-09-08.md). No allowance was reset or
increased, and the previous ledger's hash remains unchanged.

## Fixed scope and allowance

- Candidate: local opt-in `svp` profile, exactly `svp-roles-v3`.
- Model: `gpt-4.1-mini-2025-04-14`; no model fallback or reasoning-model upgrade.
- Cases: one `sofa_mc` run and one `kitchen_open` run, with their existing questions.
- New batch: **2 sessions maximum, 24 actual HTTP attempts total, 12 per session**.
  Failed requests, schema repairs, and transport retries count before submission.
- New ledger: `artifacts/svp_hybrid_2026-09-08/usage.sqlite3`.
- No automatic reruns, additional cases, allowance increases, or ledger resets.
  Two admitted sessions close the batch even if requests remain.
- Dry-run is the default: no secret read, model/embedding request, ledger creation,
  admitted session, or output directory. `--run` explicitly admits one paid case.

These are application request limits, not an OpenAI account dollar spending cap.
The existing owner-only `.env` reader is reused without sourcing the file, loading
its contents into the environment, interpolating values, or printing credentials.

## Comparison and provenance

The candidate reuses the exact historical video bytes, uniform scene boundaries,
scene captions, and normalized cached vectors used by the earlier comparison.
The cache is explicitly rebound into an isolated candidate index; a miss fails
before admission. No candidate recaptioning is allowed. Final evidence frames are
sampled normally from the candidate's selected windows.

`clean-roles-v2` and `svp-roles-v2` artifacts are **historical controls, not fresh
paired runs**. Both controls must match the candidate's fixed-input hash. The
runner records their file hashes and the closed ledger's hash/counters. It saves
candidate source hashes and exact snapshots, prompts, actual raw traces, selected
frames, usage, and results in exclusive case directories. The second candidate
run refuses changed source hashes, so it cannot silently test another revision.

These two questions are known regression cases, not unseen holdouts or a broad
accuracy benchmark. Review separately:

1. Execution and request count: does the actual typed program run within budget?
2. Retrieval: does the gift example retain its later outcome evidence without
   unnecessary early windows? Does the global kitchen question retain useful
   representative actions? Check actual temporal/cosine ranks and selection.
3. Visual grounding: inspect sampled frames against timestamps, object identities,
   actions, and answer citations. A correct main answer is not proof that every
   detail is supported; the model's own accepted review is not a correctness label.
4. Cost and observed latency: report actual tokens/HTTP attempts and local elapsed
   time, without claiming a two-case load or latency benchmark.

This separation of task-specific stage checks, traces, and visual review follows
[official OpenAI evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
The model remains the requested pinned non-reasoning snapshot documented on the
[GPT-4.1 mini model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

## Commands

```bash
.venv/bin/python scripts/evaluate_svp_hybrid.py --case sofa_mc
.venv/bin/python scripts/evaluate_svp_hybrid.py --case kitchen_open
```

Only after candidate code is frozen and the offline checks pass, add `--run` once
for each case. Do not remove an output directory or replace a ledger to retry.
