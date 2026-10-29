# NExT-QA and EgoSchema smoke test

Date: 2026-09-05

These runs validate the local QA vertical slice against real videos. They are not
dataset accuracy results: each row contains only the first locally available example.
Qwen3-VL-8B-Instruct was served locally through vLLM, while standalone retrieval used
the deterministic hash embedder because PostgreSQL was unavailable in this session.

| Dataset | Video duration | Cache state | Retrieved visual input | Latency | Result |
|---|---:|---|---|---:|---|
| NExT-QA | 12.35 s | cold | one bounded 12.35 s motion proxy | 49.93 s | incorrect (2 vs 0) |
| EgoSchema | 180 s | cold | four of nine 20 s intervals | 152.63 s | correct (3) |
| EgoSchema | 180 s | warm, tuned | four of nine 20 s intervals | 48.16 s | correct (3) |

The EgoSchema example demonstrates the intended cache behavior: scene/event analysis
is performed once, then QA inspects four selected bounded clips rather than passing the
entire three-minute source to one model request. The warm latency includes planning,
four concurrent evidence-inspection calls, answer composition, and verification.

The NExT-QA video shows two people operating hurdy-gurdy-like instruments by rotating
handles. The 8B VLM repeatedly classified the action as strumming, even with eight
frames and then the complete bounded motion proxy. This is a model-perception failure,
not a missed interval. A production system should use confidence- or disagreement-based
escalation to a larger VLM or specialist motion/object model for fine-grained mechanism
questions.

Raw JSONL outputs are written under `artifacts/evaluation/` and intentionally excluded
from source control.

## Hosted multi-step example

The featured NExT-QA example asks why a boy selects a present and carries it to a sofa.
Answering requires connecting the initial selection and movement with later intervals
where he pulls at the wrapping and opens the gift. With Qwen3-Embedding-0.6B semantic
retrieval, the system selected the correct answer, `Unwrap it`, from five choices. The
warm QA path inspected bounded evidence from later parts of the 90-second video and
completed in 44.89 seconds. Hash retrieval missed the critical intervals, demonstrating
why it remains a plumbing-only fallback.
