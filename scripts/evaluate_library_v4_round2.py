"""Second, once-only four-example library development round; no held-out mixing.

--prepare freezes the same four official examples and completed first-round
controls. Default preflight is read-only. --run permits four sessions, at most
160 HTTP attempts (40 per question), and $0.90 in a separate durable ledger.
Together with the audited first-round spend, this stays below the original $1.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_library_v4_round2_base", REPO / "scripts/evaluate_library_v4.py",
)
library = importlib.util.module_from_spec(spec)
spec.loader.exec_module(library)
base = library.base
FIRST_ROUND = REPO / "artifacts/library_v4_2026-09-08"
ROOT = REPO / "artifacts/library_v4_round2_2026-09-08"
MAX_COMBINED_USD = 1.0
base.ROOT, base.MAX_USD = ROOT, 0.90
base.HELPERS += ("scripts/evaluate_library_v4_round2.py",)
library.__doc__ = __doc__
_prepare_manifest = library.prepare_manifest


def first_round_controls():
    names = ["manifest.json", "run-config.json", "summary.json", "source-snapshots.json",
             "cost-audit.json", "usage.sqlite3", "budget.sqlite3"]
    names += [f"cases/{identifier}/{name}" for identifier in library.EXAMPLE_IDS
              for name in ("report.json", "bundle.json")]
    paths = [FIRST_ROUND / name for name in names]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise base.EvaluationError("The audited first library round must remain complete.")
    summary = json.loads((FIRST_ROUND / "summary.json").read_text())
    audit = json.loads((FIRST_ROUND / "cost-audit.json").read_text())
    budget = summary["budget"]
    if (not summary["finished"] or summary["attempted"] != 4
        or not audit["passed_audit"] or audit["actual_http_attempts"] != 62
        or budget["reserved_usd"] != 0 or budget["ambiguous_usd"] != 0
        or budget["reported_usd"] != audit["reported_usd"]
        or budget["reported_usd"] + base.MAX_USD > MAX_COMBINED_USD):
        raise base.EvaluationError("First-round accounting does not allow this bounded round.")
    return {str(path): base.file_sha(path) for path in paths}, budget["reported_usd"]


def prepare_manifest():
    manifest = _prepare_manifest()
    controls, spent = first_round_controls()
    manifest["historical_files_sha256"].update(controls)
    manifest.update(development_round=2, first_round_root=str(FIRST_ROUND),
        first_round_reported_usd=spent, round_max_usd=base.MAX_USD,
        combined_max_usd=MAX_COMBINED_USD,
        protocol="Same four previously exposed development examples, separate cold caches; "
                 "no first-round reruns or ledger resets, no held-out examples or labels "
                 "used for inference, no automatic additional rounds.")
    return manifest


def write_run_json(path, value, key):
    if path == base.ROOT / "run-config.json":
        _, spent = first_round_controls()
        value = {**value, "release_gate": library.GATE,
            "evaluation_kind": "library_development_smoke", "development_round": 2,
            "first_round_reported_usd": spent, "combined_max_usd": MAX_COMBINED_USD}
    return library.shared._write_json(path, value, key)


library.prepare_manifest, base.write_json = prepare_manifest, write_run_json


def main(argv=None):
    return library.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
