# Final development recheck

The completed nine-case smoke remains immutable: seven completed, six nonabstaining
reviewed answers, four correct; all four library answers correct. Two Answer
contract failures remain. Reported cost was $0.1880604; the independent audit is
`artifacts/packing_contract_smoke_2026-09-08/live-audit.json`.

This final recheck uses precisely the same four library examples plus
`nextqa-3359075894-q2` and `nextqa-5498331490-q2`. These are exposed development
inputs, not held-out evaluation. No further development rounds are authorized.

The new root is `artifacts/packing_contract_smoke_round2_2026-09-08`.
Its separate ledger permits six sessions, two workers, 240 actual HTTP attempts
(40 per session), and $0.30 total with $0.03 reserved per actual attempt. The
previous reported spend plus the entire new allowance is $0.4880604, below the
original $0.50 scope. Unreported usage retains its reservation and stops admission.
The pinned model is `gpt-4.1-mini-2025-04-14`; caches start cold. No scoring labels
enter the inference path. Prior raw journals, final outputs, ledgers and audit are
hash-bound in the new manifest; current source snapshots freeze at explicit launch.

The development gate requires six normal final answers with matching completed
reviews, allowing valid abstention but reporting it separately. Nonabstaining
accuracy and all-four-correct library checks remain separate. Retained previous
drafts never count as completed. No result authorizes deployment or a held-out claim.

Prepare once with `scripts/evaluate_request_contract_smoke_round2.py --prepare`.
The default invocation verifies inputs without credentials, admissions or model
calls. Only an explicit `--run` starts this cohort. `--run --resume` can admit
never-started cases only; it cannot rerun finalized or interrupted cases. Existing
manifests and ledgers are never reset.
