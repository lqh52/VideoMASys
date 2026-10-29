# Structured v5: admission and contract fixes

The public Space is unchanged. This local candidate addresses the application
failures found in the previous v4 run; it does not inherit or replace that run's
score. The model remains `gpt-4.1-mini-2025-04-14`.

## Changes

1. **Correct the local request allowance.** GPT-4.1 mini's native schema requests
   now have a conservative 65,536 input/output admission allowance, rather than
   the generic 32,768 allowance. This is an explicit policy increase, not prompt
   compression or a claim that requests use fewer tokens. Prompts, evidence,
   schemas, and images are unchanged by admission. Plain calls and other adapters
   keep the old allowance. Image count/size, output-token, prompt-character,
   request-byte, timeout and per-attempt admission limits remain enforced.
2. **Name local budget failures correctly.** A context-bound rejection reports
   `local_request_budget`, with numeric-only budget diagnostics and zero HTTP
   attempts. It is no longer presented as a provider/image failure. The same
   arithmetic is available through a read-only `request_budget` helper.
3. **Distinguish context from time order.** A directionless context need may be
   fulfilled by a real search scoped to an earlier anchor's results. It does not
   require an artificial `relate` call. Independent global searches and tags on
   the same step still do not establish a dependency. Before/after/outcome needs
   keep their strict temporal ancestry checks.
4. **Repair all missing edges together.** The tool validator reports the full
   list of missing dependencies to the one allowed repair. Structured repairs
   receive complete compact prior JSON instead of a 3,000-character slice that
   can remove final tool steps. Oversized prior JSON is explicitly omitted rather
   than supplied as misleading partial JSON; the full raw response remains in
   the trace.
5. **Normalize harmless representation differences.** Exact repeated claim
   references are deduplicated in first-occurrence order, after size/type
   validation. Option assessments are reordered by exact supplied choice text,
   only after one-per-option coverage has passed. No claims, evidence, statuses,
   selections or dependency edges are invented. Raw outputs and normalization
   metadata remain inspectable. Duplicate claim/registry IDs, unknown references,
   cycles, missing coverage and unsupported selections still fail validation.
6. **Plan evidence, not the conclusion.** For phase comparisons, the planner
   retrieves each phase's actions; the answer agent performs the comparison.
   `outcome` means a later observable event, not an analytical difference.
7. **Select conclusions as answer roots.** The answer prompt explains that
   conclusions inherit their premises through dependencies; three roots is a
   maximum, not a target. Repair diagnostics identify omitted conclusion or
   limitation IDs. The model must select justified roots itself; no controller
   promotion or invented dependency makes an invalid answer pass.

OpenAI Docs informed the model-specific allowance: the official GPT-4.1 mini
context window is 1,047,576 tokens, distinct from its 32,768 maximum output.
Our smaller allowance is an application guard, not the provider's context limit.
See [the official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Native [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and the existing semantic validators remain enabled.

## Verification

Focused offline transport, schema, pipeline, domain and retrieval checks passed.
An independent source review found no blocking issues.

Read-only replay demonstrated:

- All 15 previously rejected exact request packs serialize under the new
  allowance, retaining prompt text, schema constraints, image order and encoded
  image hashes. This verifies admission only; it does not generate answers.
- Three saved NExT-QA final tool programs and the sofa program validate unchanged
  once scoped context is treated correctly. The genuinely malformed EgoSchema
  phase chain remains invalid; its missing temporal edges are not synthesized.
- Of 14 previously invalid answer attempts, two now validate: one shuffled
  option list and one repeated-frame abstention. The other 12 semantic errors
  remain rejected. The abstention does not become a correct-answer substitute.

## Bounded live development check

The first check covered the four library examples plus the five terminal-contract
failure cases, nine cases total, each once with a fresh index. These examples are already
exposed development cases, **not a new hold-out benchmark**. No automatic retries
of failed cases, extra 40-case run, or deployment are included.

The separate runner has a $0.50 cap and a $0.03 reservation per HTTP attempt,
conservatively covering the larger allowable request. Previous helpers, ledgers,
manifests and scores stay immutable. Completion, abstention and correctness are
reported separately; the library also needs all four correct non-abstaining
answers. Passing a development smoke test does not authorize publication or
establish the requested 90%/70% generalization thresholds.

The first live check completed **7/9** analyses: six non-abstaining answers,
four correct, two wrong, and one completed abstention. **All four library
examples were correct and normally completed.** There were no local size
rejections or tool-compilation failures. Two cases stopped later at answer
validation; one retained draft was correctly excluded from completion.

Both remaining failures omitted an existing conclusion/limitation from
`answer_claim_ids`, listing only observations instead. The final small revision
clarified that answer roots select conclusions and inherit their premises;
three roots is a maximum, not a target. Repair feedback points to existing
conclusion or limitation IDs. The model must fix its roots; the controller does
not promote claims or waive validation.

The first run used **137 HTTP requests and $0.1880604**. Its independent audit
reconciled all requests, $0.03 reservations, raw/canonical evidence, source hashes,
and historical controls. The result is retained unchanged.

A final six-case development recheck covers the four library cases and the two
remaining failures, once each in a new ledger. Its $0.30 cap keeps cumulative
spending below the originally declared $0.50 allowance. This is still not a
held-out benchmark and does not authorize deployment.

### Final recheck result

**The candidate is not ready for release.** Five of six analyses completed,
including one abstention; four returned non-abstaining answers and only two were
correct. The latest library check was **2/4 correct**, not the earlier 4/4. Both
results are retained; neither is a held-out accuracy estimate.

| Case | Latest result |
| --- | --- |
| Library: sofa | Completed abstention; selected scenes 3–4, missing the earlier relevant action |
| Library: tree | Stopped at Tool-agent temporal-dependency validation |
| Library: kitchen | Correct: “C is cleaning dishes.” |
| Library: painting | Correct: “To clean the paintbrush.” |
| NExT-QA `3359075894-q2` | Completed, wrong answer; no answer-root validation error |
| NExT-QA `5498331490-q2` | Completed, wrong answer; prior omitted-limitation failure did not recur |

There were no local request-size rejections or transport failures. The two cases
targeted by the final answer-root clarification now completed, but that does not
prove their reasoning or perception is correct. One intermediate missing-need
answer error and one tool error in the cat case were repaired within the existing
single-repair policy.

The remaining terminal tree error was a real temporal-chain omission, not a
directionless context falsely rejected by the validator. The plan required
`unwrap_present → return_to_tree → tree_interaction`; the program searched after
the unwrap event, then searched within those results without a second temporal
relation anchored on the return. Its repair retagged the steps but still omitted
that relation. No controller fabricated the missing edge. The sofa run also
shows that a structurally valid program can retrieve the wrong occurrence of a
repeated action; its abstention must not be counted as a correct library answer.

The final recheck used **84 HTTP requests and $0.092474**, with no outstanding or
ambiguous charges. Combined with the first run: **$0.2805344**, below the declared
$0.50 allowance. Independent audit reconciled all 84 HTTP calls and settled
ledger tickets, 64 frozen source/snapshot hashes, 443 historical controls, and
raw/canonical evidence provenance. Audit integrity passed; performance gates did
not. No further tuning rounds, benchmark runs, or publication were
performed. The engineering fixes are implemented, but the library and completion
gates remain unmet. The public Space is unchanged.

Records:

- [Exact request admission replay](../artifacts/payload_replay_2026-09-08/report.json)
- [Nine-case live results](../artifacts/packing_contract_smoke_2026-09-08/summary.json)
- [Nine-case independent audit](../artifacts/packing_contract_smoke_2026-09-08/live-audit.json)
- [Final six-case results](../artifacts/packing_contract_smoke_round2_2026-09-08/summary.json)
- [Final independent audit](../artifacts/packing_contract_smoke_round2_2026-09-08/live-audit.json)

Historical baseline: [v4 report](structured_v4_release_2026-09-08.md).
