"""Third, bounded four-example library development round; no held-out mixing.

--prepare freezes the same reviewed examples and both completed library rounds.
Default is read-only. --run permits four sessions, 160 HTTP attempts (40 each),
and at most $0.80. Previous reported library spend plus this cap stays below $1.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_library_v4_round3_base", REPO / "scripts/evaluate_library_v4_round2.py",
)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
library, base = previous.library, previous.base
SECOND_ROUND = REPO / "artifacts/library_v4_round2_2026-09-08"
ROOT = REPO / "artifacts/library_v4_round3_2026-09-08"
base.ROOT, base.MAX_USD = ROOT, 0.80
base.HELPERS += ("scripts/evaluate_library_v4_round3.py",)
library.__doc__ = __doc__
_prepare_manifest = previous.prepare_manifest


def completed_rounds_controls():
    names = ["manifest.json", "run-config.json", "summary.json", "source-snapshots.json",
             "cost-audit.json", "usage.sqlite3", "budget.sqlite3"]
    names += [f"cases/{identifier}/{name}" for identifier in library.EXAMPLE_IDS
              for name in ("report.json", "bundle.json")]
    paths = [SECOND_ROUND / name for name in names]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise base.EvaluationError("Both earlier library rounds must remain audited and complete.")
    _, first_spent = previous.first_round_controls()
    summary = json.loads((SECOND_ROUND / "summary.json").read_text())
    audit = json.loads((SECOND_ROUND / "cost-audit.json").read_text())
    budget = summary["budget"]
    spent = first_spent + budget["reported_usd"]
    if (not summary["finished"] or summary["attempted"] != 4 or not audit["passed_audit"]
        or budget["reserved_usd"] != 0 or budget["ambiguous_usd"] != 0
        or budget["reported_usd"] != audit["reported_usd"]
        or spent != audit["combined_reported_usd"]
        or spent + base.MAX_USD > previous.MAX_COMBINED_USD):
        raise base.EvaluationError("Prior accounting does not allow this bounded third round.")
    return {str(path): base.file_sha(path) for path in paths}, spent


def prepare_manifest():
    manifest = _prepare_manifest()
    controls, spent = completed_rounds_controls()
    manifest["historical_files_sha256"].update(controls)
    manifest.update(development_round=3, second_round_root=str(SECOND_ROUND),
        previous_rounds_reported_usd=spent, round_max_usd=base.MAX_USD,
        protocol="Same four previously exposed development examples, separate cold caches; "
                 "both earlier rounds remain immutable, no ledger resets, no held-out "
                 "examples or scoring labels used for inference, no automatic extra rounds.")
    return manifest


def write_run_json(path, value, key):
    if path == base.ROOT / "run-config.json":
        _, spent = completed_rounds_controls()
        value = {**value, "release_gate": library.GATE,
            "evaluation_kind": "library_development_smoke", "development_round": 3,
            "previous_rounds_reported_usd": spent,
            "combined_max_usd": previous.MAX_COMBINED_USD}
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
