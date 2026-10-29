"""Final six-case development recheck, isolated from the completed nine-case run.

--prepare freezes the four library cases and two terminal Answer failures.
Default preflight is read-only. Only --run permits six sessions, 240 HTTP attempts
(40 per session), two workers and $0.30, with $0.03 reserved per actual attempt.
Prior reported spend plus this cap must remain below the original $0.50 budget.
No further cohorts, failed-case reruns, held-out claims or deployment are allowed.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_final_request_smoke", REPO / "scripts/evaluate_request_contract_smoke.py",
)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
PRIOR = REPO / "artifacts/packing_contract_smoke_2026-09-08"
ROOT = REPO / "artifacts/packing_contract_smoke_round2_2026-09-08"
CASE_IDS = ("nextqa-sofa", "nextqa-tree", "egoschema-kitchen", "egoschema-painting",
            "nextqa-3359075894-q2", "nextqa-5498331490-q2")
VERSION = previous.VERSION
MAX_COMBINED_USD = Decimal("0.50")
base.ROOT, base.MAX_USD = ROOT, 0.30
base.LIMITS = {"max_runs": 6, "max_requests": 240, "max_requests_per_run": 40}
base.HELPERS += ("scripts/evaluate_request_contract_smoke_round2.py",)
GATE = {**previous.GATE, "planned_cases": 6, "minimum_normally_completed_answers": 6}


def prepare_manifest():
    prior = json.loads((PRIOR / "manifest.json").read_text())
    summary = json.loads((PRIOR / "summary.json").read_text())
    audit = json.loads((PRIOR / "live-audit.json").read_text())
    budget = summary["budget"]
    spent = Decimal(str(budget["reported_usd"]))
    if (not summary["finished"] or summary["attempted"] != 9 or not audit["passed_audit"]
        or budget["reserved_usd"] != 0 or budget["ambiguous_usd"] != 0
        or budget["reported_usd"] != audit["reported_usd"]
        or spent + Decimal(str(base.MAX_USD)) > MAX_COMBINED_USD
        or audit["manifest_sha256"] != base.file_sha(PRIOR / "manifest.json")):
        raise base.EvaluationError("The audited prior run cannot admit this bounded recheck.")
    paths = [*sorted(PRIOR.glob("*.json")), *sorted((PRIOR / "cases").glob("*/*.json")),
             PRIOR / "usage.sqlite3", PRIOR / "budget.sqlite3"]
    required = [PRIOR / name for name in ("run-config.json", "source-snapshots.json")]
    required += [PRIOR / "cases" / case["id"] / name for case in prior["cases"]
                 for name in ("report.json", "bundle.json")]
    if any(path.is_symlink() or not path.is_file() for path in [*paths, *required]):
        raise base.EvaluationError("All prior audit, wire journals and final records must remain.")
    controls = {**prior["controls_sha256"],
                **{str(path): base.file_sha(path) for path in paths}}
    if not previous.controls_unchanged({"controls_sha256": controls}):
        raise base.EvaluationError("Historical run controls changed.")
    cases = {case["id"]: case for case in prior["cases"]}
    if len(cases) != 9 or any(identifier not in cases for identifier in CASE_IDS):
        raise base.EvaluationError("The fixed six-case cohort no longer matches its prior run.")
    return {**prior, "kind": "Final exposed six-case development recheck; NOT held-out",
        "development_round": 2, "previous_run_root": str(PRIOR),
        "previous_reported_usd": float(spent), "combined_max_usd": float(MAX_COMBINED_USD),
        "limits": {**base.LIMITS, "max_usd": base.MAX_USD}, "workers": 2,
        "development_gate": GATE, "publication_authorized": False,
        "protocol": "Same four library examples and the two terminal Answer failures only. "
                    "Cold indexing, unchanged model, fresh isolated ledger; prior results and "
                    "raw outputs remain immutable. No further development rounds authorized.",
        "controls_sha256": controls, "cases": [cases[identifier] for identifier in CASE_IDS]}


def load_manifest():
    path = base.ROOT / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise base.EvaluationError("Prepare the fixed six-case manifest first.")
    manifest = json.loads(path.read_text())
    if manifest != prepare_manifest() or manifest["model"] != base.GPT41_MINI_MODEL:
        raise base.EvaluationError("Frozen recheck inputs, prior records or budget changed.")
    for item in manifest["label_sources"]:
        if base.file_sha(Path(item["path"])) != item["sha256"]:
            raise base.EvaluationError("Official annotation source changed.")
    for case in manifest["cases"]:
        if (base.file_sha(Path(case["video"])) != case["video_sha256"]
            or case["options"][case["reference"]["correct_option"]]
               != case["reference"]["correct_choice"]):
            raise base.EvaluationError("Frozen media or scoring reference changed.")
    return manifest


def summarize(manifest, budget=None, guard=None):
    result = previous.summarize(manifest, budget, guard)
    rows = result["results"]
    result.update(planned=6, finished=len(rows) == 6,
                  evaluation_kind="Final exposed development recheck; not held-out")
    stable = result["frozen_inputs_unchanged"]
    completed = sum(row["completed"] for row in rows)
    result["development_gate"] = {**GATE, "completed": completed, "evaluated": result["finished"],
        "passed": bool(result["finished"] and completed == 6 and stable)}
    result["answer_metrics"]["accuracy_all_planned"] = (
        sum(row["correct"] for row in rows) / 6 if result["finished"] else None)
    result["previous_reported_usd"] = manifest["previous_reported_usd"]
    if "budget" in result:
        combined = (Decimal(str(manifest["previous_reported_usd"]))
                    + Decimal(str(result["budget"]["charged_usd"])))
        result["combined_charged_usd"] = float(combined)
        result["combined_budget_satisfied"] = combined <= MAX_COMBINED_USD
        result["development_gate"]["passed"] &= result["combined_budget_satisfied"]
        result["library_gate"]["passed"] &= result["combined_budget_satisfied"]
    result["publication_authorized"] = False
    return result


def write_run_json(path, value, key):
    if path == base.ROOT / "run-config.json":
        manifest = load_manifest()
        value = {**value, "development_gate": GATE, "development_round": 2,
            "previous_reported_usd": manifest["previous_reported_usd"],
            "combined_max_usd": float(MAX_COMBINED_USD), "request_reservation_usd": 0.03,
            "structured_context_allowance": 65_536, "explicit_launch_requested": True,
            "evaluation_kind": "final_exposed_development_recheck", "publication_authorized": False}
    return previous.shared._write_json(path, value, key)


base.load_manifest, base.summarize, base.write_json = load_manifest, summarize, write_run_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=(2,), default=2)
    args = parser.parse_args(argv)
    if base.PROMPT_VERSION != VERSION:
        raise base.EvaluationError("This recheck is pinned to structured-evidence-v5.")
    if args.resume and not args.run:
        parser.error("--resume requires --run and only admits never-started cases")
    if args.prepare:
        from replay_rejected_packs import write_exclusive

        write_exclusive(base.ROOT / "manifest.json", prepare_manifest())
    manifest = load_manifest()
    if not args.run:
        print(json.dumps({"dry_run": True, "cases": 6, "model": base.GPT41_MINI_MODEL,
            "prompt_version": VERSION, "limits": base.LIMITS, "max_usd": base.MAX_USD,
            "previous_reported_usd": manifest["previous_reported_usd"],
            "combined_max_usd": float(MAX_COMBINED_USD), "request_reservation_usd": 0.03,
            "development_gate": GATE, "model_called": False, "credential_read": False,
            "publication_authorized": False}, indent=2))
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
