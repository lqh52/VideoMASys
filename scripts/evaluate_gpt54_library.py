"""Four fresh, full-pipeline GPT-5.4 library cases; local development only.

--prepare freezes inputs without credentials/inference. Default is read-only
preflight; --run explicitly admits at most 4 sessions, 100 HTTP attempts (25 per
session), and $2 under a separate persistent ledger. No historical cache reuse,
public settings changes, or held-out accuracy claim. Prices checked 2026-09-08:
https://developers.openai.com/api/docs/models/gpt-5.4
"""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from videomasys.zerogpu.openai import GPT54_MODEL, OpenAI54VLM, OpenAIVLM  # noqa: E402
from videomasys.zerogpu.remote import MAX_OUTPUT_TOKENS  # noqa: E402

# Isolated helper instances: no old runner globals, ledgers or records are changed.
spec = importlib.util.spec_from_file_location(
    "_gpt54_library_shared", REPO / "scripts/evaluate_library_v4.py",
)
library = importlib.util.module_from_spec(spec)
spec.loader.exec_module(library)
base, shared = library.base, library.shared
spec = importlib.util.spec_from_file_location(
    "_gpt54_library_accounting", REPO / "scripts/benchmark_accounting.py",
)
accounting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accounting)

ROOT = REPO / "artifacts/gpt54_library_2026-09-08"
VERSION = "structured-evidence-v6"
PRICES_PER_MILLION_USD = {"input": 2.5, "cached_input": 0.25, "output": 15}
# $0.00000005 units represent even a single cached token exactly ($0.00000025).
accounting.UNITS_PER_USD = 20_000_000
accounting.RESERVATION_UNITS = 3_800_000
accounting.RESERVATION_USD = 0.19
accounting.MAX_CONTEXT_BOUND = 65_536


def _cost(usage):
    """Keep the audited malformed-usage policy, using the full model's exact rates."""
    if not isinstance(usage, dict):
        return accounting.RESERVATION_UNITS, False
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details", {})
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
    count = accounting._count
    complete = (
        count(prompt) and count(output) and count(cached) and cached <= prompt
        and ("total_tokens" not in usage or (
            count(usage["total_tokens"]) and usage["total_tokens"] == prompt + output
        ))
    )
    if complete:
        return 50 * (prompt - cached) + 5 * cached + 300 * output, True
    known = (50 * prompt if count(prompt) else 0) + (300 * output if count(output) else 0)
    return max(accounting.RESERVATION_UNITS, known), False


accounting._cost = _cost


class Budget(accounting.Budget):
    def __init__(self, path, max_usd=2.0):
        # 65,536 input + 1,600 output tokens cost at most $0.18784.
        if 50 * accounting.MAX_CONTEXT_BOUND + 300 * MAX_OUTPUT_TOKENS \
                > accounting.RESERVATION_UNITS:
            raise ValueError("GPT-5.4 adapter limits exceed the fixed request reservation.")
        super().__init__(path, max_usd=max_usd)


class LoggedVLM(OpenAI54VLM, accounting.BenchmarkVLM):
    """Full-model transport plus existing per-attempt accounting and wire provenance."""

    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.output, self.sequence = output, 0

    def _generation_parameters(self, tokens):
        # The provenance helper historically subclasses 4.1-mini. Bypass only
        # that ancestor's parameter override, retaining the full adapter's routing,
        # context limit and exact reported-model checks.
        return {**OpenAIVLM._generation_parameters(self, tokens), "temperature": 0}

    def _generate(self, prompt, paths, tokens, *, schema=None):
        self.sequence += 1
        record = {"prompt": prompt, "images": list(paths), "schema": schema,
                  "max_output_tokens": tokens, "started_at": base.utc_now()}
        base.write_json(self.output / f"call-{self.sequence:03d}-started.json", record,
                        self._token)
        try:
            value = super()._generate(prompt, paths, tokens, schema=schema)
            record["raw_output"] = value
            return value
        finally:
            record.update(finished_at=base.utc_now(),
                          transport=self.calls[-1] if self.calls else {})
            base.write_json(self.output / f"call-{self.sequence:03d}.json", record, self._token)


_prepare_manifest = library.prepare_manifest
_write_json = library.write_run_json


def prepare_manifest():
    # The shared helper validates official annotations and explicit UI aliases.
    # References remain scorer-only: run_case passes only question/options/media.
    manifest = _prepare_manifest()
    manifest.update(
        model=GPT54_MODEL, reasoning_effort="none", intended_prompt_version=VERSION,
        purpose="Previously exposed library GPT-5.4 development comparison; not held-out.",
        prices_per_million_usd=PRICES_PER_MILLION_USD,
        reservation_usd=accounting.RESERVATION_USD,
        limits=base.LIMITS, max_usd=base.MAX_USD,
        cold_cache_policy="New cache per question; fresh GPT-5.4 captions, planning, "
        "inspection, answer and review. No previous model outputs or index seeds reused.",
    )
    return manifest


def write_run_json(path, value, key):
    if path == base.ROOT / "run-config.json":
        value = {**value, "evaluation_kind": "gpt54_library_development_comparison",
                 "reasoning_effort": "none", "prices_per_million_usd": PRICES_PER_MILLION_USD,
                 "reservation_usd": accounting.RESERVATION_USD,
                 "accounting_units_per_usd": accounting.UNITS_PER_USD,
                 "public_deployment": False, "all_roles_fresh": True}
        # Use the immutable writer directly so the historical library label does
        # not overwrite this batch's evaluation_kind.
        return shared._write_json(path, value, key)
    return _write_json(path, value, key)


def final_state(bundle):
    """Match the explicit v6 reviewed pair, including failed optional polish."""
    if (bundle.get("final_answer_complete") is not True
        or bundle.get("review_complete") is not True or bundle.get("retained_previous_draft")):
        return False
    pair, trace = bundle.get("reviewed_pair", {}), bundle.get("trace", [])
    ai, ri = pair.get("answer_trace_index"), pair.get("review_trace_index")
    if not (type(ai) is int and type(ri) is int and 0 <= ai < ri < len(trace)):
        return False
    answer, review = trace[ai], trace[ri]
    if (answer.get("stage") != "Answer" or review.get("stage") != "Verifier"
        or answer.get("status") != "complete" or review.get("status") != "complete"
        or answer.get("revision") != review.get("revision")
        or answer.get("output") != bundle.get("answer")
        or answer.get("structured_output") != bundle.get("evidence_graph", {}).get("reasoning")
        or review.get("structured_output")
           != bundle.get("verification", {}).get("direct_assessments")):
        return False
    expected = deepcopy(review["output"])
    later = [step for step in trace[ri + 1:] if step.get("stage") in ("Answer", "Verifier")]
    if bundle.get("retained_reviewed_answer"):
        if (not bundle.get("explanation_repair_failed") or not expected.get("accepted")
            or expected.get("revision_scope") != "explanation"
            or bundle.get("status") != "unverified"
            or not any(step.get("status") == "failed" for step in later)):
            return False
        expected.update(needs_revision=False, explanation_repair_failed=True,
                        advisories=[*expected["advisories"],
                            "Explanation refinement did not complete; "
                            "the previously reviewed answer is unchanged."])
    elif later:
        return False
    return expected == bundle.get("verification")


library.ROOT, library.VERSION = ROOT, VERSION
library.__doc__ = __doc__
base.ROOT, base.VERSION = ROOT, VERSION
# This historical variable names the selected model in orchestration/metadata;
# the actual adapter above, request body and reported-model check are full GPT-5.4.
base.GPT41_MINI_MODEL = GPT54_MODEL
base.LIMITS = {"max_runs": 4, "max_requests": 100, "max_requests_per_run": 25}
base.MAX_USD = 2.0
base.Budget, base.LoggedVLM = Budget, LoggedVLM
base.HELPERS += ("scripts/evaluate_gpt54_library.py",)
library.prepare_manifest = prepare_manifest
base.write_json = write_run_json
shared.final_state = final_state


def main(argv=None):
    # Reuse the audited exclusive preparation, no-secret default preflight,
    # source snapshots, once-only cases, final Answer/Verifier pairing and gates.
    return library.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
