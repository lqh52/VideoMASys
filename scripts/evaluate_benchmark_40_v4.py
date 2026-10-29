"""New fixed 20+20 v4 release gate, reusing audited accounting and orchestration.

No earlier script or batch is modified. Default preflight reads no credentials and
performs no admission/inference. --run permits only this new $4 batch; --resume
continues never-started cases, never failures or retained drafts.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import sys
import time
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]

from prepare_benchmark_40 import annotations  # noqa: E402
from prepare_benchmark_40_v4 import RELEASE_GATE, ROOT, SEED, VERSION  # noqa: E402

# Isolate the audited runner's configuration/hooks: importing this file never
# changes the original module's public defaults or touches its old ledger.
spec = importlib.util.spec_from_file_location(
    "_benchmark40_v4_base", REPO / "scripts/evaluate_benchmark_40.py",
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT, base.VERSION = ROOT, VERSION
base.HELPERS += ("scripts/prepare_benchmark_40_v4.py", "scripts/evaluate_benchmark_40_v4.py")
_load_manifest, _summarize, _write_json = base.load_manifest, base.summarize, base.write_json


def controls_unchanged(manifest):
    return all(not Path(path).is_symlink() and Path(path).is_file()
               and base.file_sha(Path(path)) == digest
               for path, digest in manifest["historical_files_sha256"].items())


def frozen_inputs_unchanged():
    """A release requires the exact source and manifest bytes admitted at launch."""
    try:
        config_path, manifest_path = base.ROOT / "run-config.json", base.ROOT / "manifest.json"
        if any(path.is_symlink() or not path.is_file() for path in (config_path, manifest_path)):
            return False
        config = json.loads(config_path.read_text())
        return (config["source_sha256"] == base.sources()
                and config["manifest_sha256"] == base.file_sha(manifest_path))
    except (OSError, ValueError, KeyError):
        return False


def load_manifest():
    manifest = _load_manifest()
    if (manifest.get("seed") != SEED or manifest.get("release_gate") != RELEASE_GATE
        or manifest.get("intended_prompt_version") != VERSION
        or manifest.get("intended_prompt_profile") != "structured"):
        raise base.EvaluationError("The new manifest does not match the fixed v4 protocol.")
    if not controls_unchanged(manifest):
        raise base.EvaluationError("Prior benchmark controls changed; do not overwrite them.")
    prior = json.loads(Path(manifest["previous_benchmark_manifest"]).read_text())
    ids = {case["video_id"] for case in manifest["cases"]}
    hashes = {case["video_sha256"] for case in manifest["cases"]}
    excluded = {value for values in manifest["excluded_video_ids"].values() for value in values}
    if (len(ids) != 40 or len(hashes) != 40 or ids & excluded
        or ids & {c["video_id"] for c in prior["cases"]}
        or hashes & {c["video_sha256"] for c in prior["cases"]}):
        raise base.EvaluationError("New cases overlap exclusions or duplicate a prior video.")
    sources = manifest["label_sources"]
    if any(base.file_sha(Path(item["path"])) != item["sha256"] for item in sources):
        raise base.EvaluationError("Official annotation sources changed.")
    nextqa = next(Path(item["path"]).parent.parent for item in sources
                  if Path(item["path"]).name == "val.csv")
    ego = next(Path(item["path"]).parent for item in sources
               if Path(item["path"]).name == "subset_answers.json")
    official, _ = annotations(nextqa, ego)
    official = {row["id"]: row for rows in official.values() for row in rows}
    for case in manifest["cases"]:
        row = official.get(case["id"], {})
        if (row.get("question") != case["question"] or row.get("options") != case["options"]
            or row.get("label") != case["reference"]["correct_option"]):
            raise base.EvaluationError("The cohort does not match official annotations.")
    return manifest


def final_state(bundle):
    """A matching final answer/review is required; never credit a retained old draft."""
    if (bundle.get("final_answer_complete") is not True
        or bundle.get("review_complete") is not True or bundle.get("retained_previous_draft")):
        return False
    answers = [(i, stage) for i, stage in enumerate(bundle.get("trace", []))
               if stage.get("stage") == "Answer"]
    reviews = [(i, stage) for i, stage in enumerate(bundle.get("trace", []))
               if stage.get("stage") == "Verifier"]
    if not answers or not reviews:
        return False
    ai, answer = answers[-1]
    ri, review = reviews[-1]
    return (ai < ri and answer.get("status") == review.get("status") == "complete"
            and answer.get("revision") == review.get("revision")
            and answer.get("output") == bundle.get("answer")
            and review.get("output") == bundle.get("verification"))


def nonabstaining(bundle):
    answer, options = bundle.get("answer", {}), bundle.get("options", [])
    index = answer.get("selected_option")
    return (type(index) is int and 0 <= index < len(options)
            and answer.get("selected_choice") == options[index]
            and isinstance(answer.get("answer"), str) and bool(answer["answer"].strip())
            and bundle.get("resolution") != "insufficient_evidence")


def run_case(case, key, guard, budget, embed, frozen):
    """Same cold-index/session path as v3; only completion/scoring gate is strengthened."""
    if base.sources() != frozen:
        raise base.EvaluationError("Sources changed; refusing a mixed-version benchmark.")
    output = base.ROOT / "cases" / case["id"]
    output.mkdir(mode=0o700)
    base.write_json(output / "started.json", {"id": case["id"], "started_at": base.utc_now()}, key)
    started, phase, failure = time.monotonic(), "Preparation", None
    prepared_seconds = index_seconds = qa_seconds = None
    index_call_count, qa_started = 0, None
    bundle = {"trace": [], "question": case["question"], "options": case["options"]}
    try:
        bundle.update(base.prepare_video(Path(case["video"]), output / "cache"))
        if bundle["key"] != case["video_sha256"]:
            raise base.EvaluationError("Video bytes differ from the frozen manifest.")
        prepared_seconds = time.monotonic() - started
        factory = partial(base.LoggedVLM, budget=budget, case_id=case["id"], output=output)
        pipeline = base.RemotePipeline(provider="openai", token=key, guard=guard, embed=embed,
                                       prompt_profile="structured", vlm_factory=factory)
        with pipeline.analysis_session(bundle) as active:
            phase, index_started = "Indexing", time.monotonic()
            active.index(bundle)
            if bundle["cached"]:
                raise base.EvaluationError("This benchmark requires a fresh video cache.")
            active.save_index(bundle)
            index_seconds = time.monotonic() - index_started
            index_call_count = len(bundle["inference"]["calls"])
            bundle["trace"].append({"stage": "Video cache", "status": "complete",
                "cache_hit": False, "scene_count": len(bundle["scenes"]),
                "scenes": bundle["scenes"], "source": "fresh captions and CPU embeddings",
                "latency_ms": round(index_seconds * 1000)})
            base.atomic_json(output / "checkpoint.json", bundle, key)
            phase, qa_started = "QA", time.monotonic()
            iterator = active.qa(bundle, case["question"], case["options"])
            try:
                for bundle in iterator:
                    base.atomic_json(output / "checkpoint.json", bundle, key)
            finally:
                iterator.close()
                qa_seconds = time.monotonic() - qa_started
    except Exception as exc:
        failure = base.safe_failure(exc)
        bundle["error"] = failure[1]
        if qa_started is not None:
            qa_seconds = time.monotonic() - qa_started
    unchanged = base.sources() == frozen
    if not unchanged:
        failure = ("source_drift", "Sources changed during inference; result is ineligible.")
    completed = failure is None and final_state(bundle)
    if failure is None and not completed:
        failure = ("incomplete_final", "No normally completed matching final answer/review; "
                   "retained earlier drafts are ineligible.")
    eligible = completed and nonabstaining(bundle)
    answer, trace = bundle.get("answer", {}), bundle["trace"]
    selected = answer.get("selected_option")
    calls = bundle.get("inference", {}).get("calls", [])
    if phase == "Indexing":
        index_call_count = len(calls)
    choice = next((s.get("output", {}).get("selected_choice") for s in trace
                   if s.get("stage") == "Choice selection"), None)
    report = {
        "id": case["id"], "dataset": case["dataset"], "split": case["split"],
        "question_type": case.get("question_type"), "completed": completed,
        "release_eligible_answer": eligible, "selected_option": selected,
        "selected_choice": answer.get("selected_choice"),
        "correct_option": case["reference"]["correct_option"],
        "correct": eligible and selected == case["reference"]["correct_option"],
        "provisional_choice_correct": choice == case["reference"]["correct_choice"],
        "answer": answer.get("answer"), "status": bundle.get("status"),
        "review_accepted": bundle.get("verification", {}).get("accepted"),
        "review_complete": bundle.get("review_complete") is True,
        "final_answer_complete": bundle.get("final_answer_complete") is True,
        "retained_previous_draft": bool(bundle.get("retained_previous_draft")),
        "resolution": bundle.get("resolution"), "error": failure,
        "failure_phase": phase if failure else None,
        "last_stage": trace[-1].get("stage") if trace else phase,
        "duration_seconds": case["duration_seconds"], "cold_cache": not bundle.get("cached"),
        "preparation_seconds": prepared_seconds, "index_seconds": index_seconds,
        "qa_seconds": qa_seconds, "total_seconds": time.monotonic() - started,
        "selected_scenes": [s["index"] for s in bundle.get("selected", [])],
        "selected_seconds": sum(s["end"] - s["start"] for s in bundle.get("selected", [])),
        "format_repairs": sum(s.get("repair_count", 0) if s.get("stage") == "Evidence inspection"
                              else max(0, len(s.get("attempts", [])) - 1) for s in trace),
        "answer_revisions": max(0, sum(s.get("stage") == "Answer" for s in trace) - 1),
        "usage": base.tokens_from_calls(calls),
        "index_usage": base.tokens_from_calls(calls[:index_call_count]),
        "qa_usage": base.tokens_from_calls(calls[index_call_count:]),
        "sources_unchanged": unchanged, "finished_at": base.utc_now(),
    }
    base.write_json(output / "bundle.json", bundle, key)
    base.write_json(output / "report.json", report, key)
    print(json.dumps({k: report[k] for k in (
        "id", "completed", "release_eligible_answer", "correct", "status", "error",
    )}), flush=True)
    return report


def summarize(manifest, budget=None, guard=None):
    result = _summarize(manifest, budget, guard)
    eligible = sum(row.get("release_eligible_answer", False) for row in result["results"])
    correct = sum(row["correct"] for row in result["results"])
    result["release_gate"] = {**RELEASE_GATE, "nonabstaining_reviewed_answers": eligible,
        "correct": correct, "evaluated": result["finished"],
        "passed": bool(result["finished"] and eligible >= 36 and correct >= 28)}
    result["frozen_inputs_unchanged"] = (frozen_inputs_unchanged()
        and all(row.get("sources_unchanged") is True for row in result["results"]))
    result["release_gate"]["passed"] &= result["frozen_inputs_unchanged"]
    for dataset, values in result["datasets"].items():
        values["nonabstaining_reviewed_answers"] = sum(
            row.get("release_eligible_answer", False) for row in result["results"]
            if row["dataset"] == dataset)
    if result["finished"]:
        result["historical_controls_unchanged"] = controls_unchanged(manifest)
        result["release_gate"]["passed"] &= result["historical_controls_unchanged"]
    return result


def write_run_json(path, value, key):
    if path == base.ROOT / "run-config.json":
        value = {**value, "release_gate": RELEASE_GATE, "selection_seed": SEED}
    return _write_json(path, value, key)


base.load_manifest, base.run_case = load_manifest, run_case
base.summarize, base.write_json = summarize, write_run_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=(2,), default=2)
    args = parser.parse_args(argv)
    if base.PROMPT_VERSION != VERSION:
        raise base.EvaluationError("This new benchmark is pinned to structured-evidence-v4.")
    manifest = load_manifest()
    if not args.run:
        print(json.dumps({"dry_run": True, "cases": 40, "seed": SEED,
            "model": base.GPT41_MINI_MODEL, "prompt_version": VERSION,
            "source_sha256": base.sources(), "limits": base.LIMITS, "max_usd": base.MAX_USD,
            "release_gate": RELEASE_GATE, "credential_read": False, "model_called": False,
            "historical_controls_unchanged": True}, indent=2))
        return 0
    with (base.ROOT / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = base.run(manifest, workers=args.workers, resume=args.resume)
        result = json.loads((base.ROOT / "summary.json").read_text())
        return code if code else (0 if result["release_gate"]["passed"] else 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
