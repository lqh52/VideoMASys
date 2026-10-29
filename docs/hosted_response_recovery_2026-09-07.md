# Hosted response failures

## Incident and diagnosis

The public app reported “The hosted response was invalid. No answer was
substituted.” The Space was still `RUNNING` on `cpu-basic`, at commit
`0bfe72751dbf981e6f18a7e6c24dcb019e53be32`. Its server logs contained startup
messages but no response-shape diagnostics. The exact failing video, stage and
provider payload were not available during the initial investigation.

This specific message originates in the hosted HTTP response adapter, before
the agent's JSON/schema validation. It can mean an empty completion, malformed
response envelope, or oversized response. The old adapter discarded useful
finish-reason and usage metadata when rejecting a response. It is not evidence
that retrieval failed or that the inference credit was exhausted.

One bounded diagnostic replay used the existing real first-scene caption prompt
and four frames from the sofa example. It made exactly one request to the same
Qwen3-VL-8B / Featherless route and succeeded: HTTP 200, one string completion,
`finish_reason=stop`, 135 characters of valid JSON, approximately 2.9 seconds.
This did **not** reproduce the user's original failure or verify a full analysis.

## Recovery boundaries

- Retain safe response shape, finish reason and allowlisted token counts even
  when parsing fails. Server warnings never include response bodies, prompts,
  image data, upstream error messages, or credentials.
- Permit one retry only for a fully received, structurally valid, empty assistant
  string completion explicitly finished with `stop`, without reasoning, refusal,
  tool calls or an error payload. It uses the existing two-attempt ceiling,
  deadline and request allowance. This can incur one additional billed request.
- Do not coerce reasoning, tool arguments or arbitrary content objects into
  answers. Malformed envelopes, refusals and ambiguous timeouts do not trigger
  this recovery. Real nonempty output still goes through the existing agent
  validators and bounded JSON repair.
- Show the failed pipeline stage in the error banner and retain partial results
  and provider diagnostics in the raw trace. Close the analysis session before
  yielding the final failure view.
- Keep the same model, CPU hardware, shared five-run/120-request allowance and
  32-request-per-analysis limit. No automatic provider or larger-model fallback.

The HF response contract describes a string assistant completion for this
non-streaming endpoint: [chat completion API specification](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion#response).

The application ledger remains ephemeral, not a durable account-wide spending
cap. A replacement Space container can reset its counters; code deployment must
not be described as preserving accumulated usage across container replacement.

## Automated verification

The full suite passed: **618 tests**, with relevant Ruff checks clean. This
includes 92 hosted-adapter tests, a real adapter/mock-transport integration with
the shared SQLite usage guard (one-request denial versus two-request recovery),
and UI failure-view cleanup/trace tests. Tests do not contact inference providers.

Covered cases include an empty response followed by real text, two empty
responses exhausting the existing attempt limit, 503 followed by empty response
without a third request, denial before recovery transport, malformed/oversized
envelopes, null error fields, refusals, tool calls, reasoning-only responses,
unknown finish reasons, metadata redaction, and retention of actual truncated
text for the existing core JSON repair. Unknown non-null assistant message
fields also block the empty-response retry, without changing acceptance of real
nonempty text or logging those fields. The browser smoke helper now captures
the full displayed raw trace on success or failure without submitting a second
analysis, and separates completed QA from scrolling/trace-capture checks.
Two local Gradio fixture checks verified full JSON capture, clipboard-method
restoration and accordion restoration without any model calls or external
browser requests. Those authored fixtures are only UI tests, not model evidence.

## Deployed verification

Final public commit: `48b1b3b9ba52dae6fb9b31e9abe79f3287a85c3d`.
The HF runtime confirmed `RUNNING`, that exact runtime SHA, and both current and
requested hardware `cpu-basic`. No model, hardware, token, subscription, refill
or configured allowance change accompanied these code-only updates.

The first browser check counted library cards before Gradio hydration completed
and exited with **no analysis submitted**. The helper now waits for a visible
card before counting. That check is retained in
`artifacts/hosted_response_recovery_2026-09-07/anonymous_sofa/`.

Exactly one anonymous full analysis was then submitted through the HF wrapper:
the library sofa example, including new scene captions/indexing. Artifacts:
`artifacts/hosted_response_recovery_2026-09-07/anonymous_sofa_live/`.

- Completed in **50.3 seconds**; 15 model calls / 15 HTTP attempts, all HTTP 200
  and `response_issue=none`. No adapter recovery was needed. The ledger reported
  one run and 15 requests used, with the unchanged five-run/120-request limits.
- The full displayed JSON trace was captured. Actual pipeline stages completed
  through the reviewer, including one answer revision. No answer was substituted.
- The answer selected **Unwrap it**, but the final reviewer rejected the draft;
  the app correctly displayed **Needs review / Draft · not verified**. Retrieval
  chose scenes 1/2/4, omitting scene 3 (54–72 seconds) that prior successful runs
  selected. This is an evidence-selection/answer-quality limitation, **not** a
  successful accuracy or verification result. No prompts/retrieval rules were
  changed to force the expected answer in this incident fix.
- All three expand/collapse cycles and footer reachability passed, with no
  browser errors. The bottom screenshot was visually inspected; all 18 sampled
  evidence thumbnails were present.

The reported original response failure was **not reproduced**. The deployment
adds bounded recovery for one narrowly defined empty-response case and preserves
diagnostics for future incidents; it is not proof that every provider failure is
resolved. No second full analysis was submitted to chase a passing review.
