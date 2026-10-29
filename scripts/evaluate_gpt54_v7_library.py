"""One fresh four-case GPT-5.4 v7 library comparison; never changes the public app.

--prepare freezes the new manifest without inference; default is read-only
preflight. --run retains the existing once-only, cold-evidence execution and
limits: $2 total, $0.19 per attempted request, 4 sessions, 100 HTTP attempts,
25 attempts per session. Previous runs and ledgers are never reused or rewritten.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location(
    "_gpt54_v7_library_base", REPO / "scripts/evaluate_gpt54_library.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

ROOT = REPO / "artifacts/gpt54_v7_library_2026-09-08"
VERSION = "structured-evidence-v7"
base = runner.base
for configured in (runner, runner.library, base):
    configured.ROOT, configured.VERSION = ROOT, VERSION
runner.library.__doc__ = __doc__
base.HELPERS += ("scripts/evaluate_gpt54_v7_library.py",)


def main(argv=None):
    return runner.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
