"""Explicit, once-only nine-case development smoke; never a held-out evaluation.

Default preflight reads no key and makes no calls. --run permits only this prepared
cohort, two workers, nine sessions, 360 actual HTTP attempts (40 per session), and
a $0.50 conservative ledger using a $0.03 reservation per attempt. No deployment.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from replay_rejected_packs import SMOKE_LIMITS, SMOKE_ROOT, smoke_manifest  # noqa: E402


def isolated(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = isolated("_request_smoke_runner", "evaluate_benchmark_40_v4.py")
accounting = isolated("_request_smoke_accounting", "benchmark_accounting.py")
accounting.RESERVATION_UNITS = 300_000
accounting.RESERVATION_USD = 0.03
accounting.MAX_CONTEXT_BOUND = 65_536
base = shared.base
VERSION = "structured-evidence-v5"
base.ROOT, base.VERSION, base.MAX_USD = SMOKE_ROOT, VERSION, 0.50
base.LIMITS = {key: SMOKE_LIMITS[key] for key in
               ("max_runs", "max_requests", "max_requests_per_run")}
base.Budget = accounting.Budget
base.HELPERS += ("scripts/replay_rejected_packs.py", "scripts/evaluate_request_contract_smoke.py")
GATE = {"planned_cases": 9, "minimum_normally_completed_answers": 9,
        "policy": "Every case must finish a matching answer and review. Valid abstention is "
                  "a completed execution, reported separately from nonabstaining accuracy. "
                  "This is a development regression gate, not publication authorization."}


class LoggedVLM(accounting.BenchmarkVLM):
    """Same durable call journals, with the isolated larger accounting reservation."""

    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.output, self.sequence = output, 0

    def _generate(self, prompt, paths, tokens, *, schema=None):
        self.sequence += 1
        record = {"prompt": prompt, "images": list(paths), "schema": schema,
                  "max_output_tokens": tokens, "started_at": base.utc_now()}
        base.write_json(self.output / f"call-{self.sequence:03d}-started.json", record, self._token)
        try:
            value = super()._generate(prompt, paths, tokens, schema=schema)
            record["raw_output"] = value
            return value
        finally:
            record.update(finished_at=base.utc_now(),
                          transport=self.calls[-1] if self.calls else {})
            base.write_json(self.output / f"call-{self.sequence:03d}.json", record, self._token)


base.LoggedVLM = LoggedVLM


def load_manifest():
    path = base.ROOT / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise base.EvaluationError("Prepare the fixed nine-case development manifest first.")
    manifest = json.loads(path.read_text())
    if manifest != smoke_manifest() or manifest["limits"] != SMOKE_LIMITS:
        raise base.EvaluationError("Prepared development inputs or controls changed.")
    for case in manifest["cases"]:
        if (not Path(case["video"]).is_file()
            or base.file_sha(Path(case["video"])) != case["video_sha256"]
            or case["options"][case["reference"]["correct_option"]]
               != case["reference"]["correct_choice"]):
            raise base.EvaluationError("Frozen media or scoring reference does not match.")
    return manifest


def controls_unchanged(manifest):
    return all(Path(path).is_file() and not Path(path).is_symlink()
               and base.file_sha(Path(path)) == value
               for path, value in manifest["controls_sha256"].items())


def summarize(manifest, budget=None, guard=None):
    result = shared._summarize(manifest, budget, guard)
    rows = result["results"]
    result.update(planned=9, finished=len(rows) == 9,
                  evaluation_kind="Exposed development regression smoke; not held-out")
    for dataset, values in result["datasets"].items():
        planned = sum(case["dataset"] == dataset for case in manifest["cases"])
        values["planned"] = planned
        values["accuracy_all_planned"] = (values["correct"] / planned
            if values["attempted"] == planned else None)
        values["nonabstaining_reviewed_answers"] = sum(
            row.get("release_eligible_answer", False) for row in rows if row["dataset"] == dataset)
    stable = (shared.frozen_inputs_unchanged() and controls_unchanged(manifest)
              and all(row.get("sources_unchanged") is True for row in rows))
    completed = sum(row["completed"] for row in rows)
    eligible = sum(row.get("release_eligible_answer", False) for row in rows)
    correct = sum(row["correct"] for row in rows)
    result["frozen_inputs_unchanged"] = stable
    result["development_gate"] = {**GATE, "completed": completed, "evaluated": result["finished"],
                                  "passed": bool(result["finished"] and completed == 9 and stable)}
    library_ids = {case["id"] for case in manifest["cases"][:4]}
    library_rows = [row for row in rows if row["id"] in library_ids]
    result["answer_metrics"] = {"nonabstaining_reviewed_answers": eligible, "correct": correct,
        "completed_abstentions": sum(row["completed"] and row["selected_option"] is None
                                     for row in rows),
        "accuracy_all_planned": correct / 9 if result["finished"] else None}
    result["library_gate"] = {"planned": 4, "finalized": len(library_rows),
        "nonabstaining_reviewed_answers": sum(row.get("release_eligible_answer", False)
                                               for row in library_rows),
        "correct": sum(row["correct"] for row in library_rows),
        "passed": bool(len(library_rows) == 4 and all(row["correct"]
                       and row.get("release_eligible_answer") for row in library_rows) and stable),
        "publication_authorized": False}
    return result


def write_run_json(path, value, key):
    if path == base.ROOT / "run-config.json":
        value = {**value, "development_gate": GATE, "request_reservation_usd": 0.03,
            "structured_context_allowance": 65_536, "explicit_launch_requested": True,
            "evaluation_kind": "exposed_development_smoke", "publication_authorized": False}
    return shared._write_json(path, value, key)


base.load_manifest, base.summarize, base.write_json = load_manifest, summarize, write_run_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=(2,), default=2)
    args = parser.parse_args(argv)
    if base.PROMPT_VERSION != VERSION:
        raise base.EvaluationError("This regression smoke is pinned to structured-evidence-v5.")
    manifest = load_manifest()
    if not args.run:
        print(json.dumps({"dry_run": True, "cases": 9, "model": base.GPT41_MINI_MODEL,
            "prompt_version": VERSION, "limits": base.LIMITS, "max_usd": base.MAX_USD,
            "request_reservation_usd": 0.03, "development_gate": GATE,
            "model_called": False, "credential_read": False, "publication_authorized": False},
            indent=2))
        return 0
    with (base.ROOT / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = base.run(manifest, workers=args.workers, resume=args.resume)
        result = json.loads((base.ROOT / "summary.json").read_text())
        return code if code else (0 if result["development_gate"]["passed"] else 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
