"""Two-case downstream-only diagnostic; saved evidence/candidate, new Answer+Verifier.

--prepare freezes inputs. Default preflight reads no credentials and calls no model.
Only --run admits two sessions, at most eight HTTP attempts each, sixteen total,
and $0.15 with $0.03 reserved per attempt. No retries of cases or resume/reset.
This is neither a full-pipeline run nor a benchmark or deployment authorization.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import sys
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def isolated(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = isolated("_downstream_replay_helpers", "scripts/evaluate_request_contract_smoke.py")
base = shared.base
ROOT = REPO / "artifacts/structured_v6_answer_replay_2026-09-08"
PRIOR = REPO / "artifacts/packing_contract_smoke_round2_2026-09-08"
CASE_IDS = ("nextqa-sofa", "nextqa-5498331490-q2")
VERSION = "structured-evidence-v6"
LIMITS = {"max_runs": 2, "max_requests": 16, "max_requests_per_run": 8}
MAX_USD = 0.15
base.ROOT = ROOT
base.HELPERS += ("scripts/replay_structured_v6_answers.py",)
base.write_json = shared.shared._write_json
RESTORE_NOTE = (
    "Explanation refinement did not complete; the previously reviewed answer is unchanged."
)


def read_json(path):
    if path.is_symlink() or not path.is_file():
        raise base.EvaluationError("Required immutable replay input is missing or not regular.")
    return json.loads(path.read_text())


def prepare_manifest():
    prior = read_json(PRIOR / "manifest.json")
    audit = read_json(PRIOR / "live-audit.json")
    if not audit["passed_audit"] or not audit["finished"]:
        raise base.EvaluationError("The prior six-case run must be completed and audited.")
    cases = {case["id"]: case for case in prior["cases"]}
    paths = [*sorted(PRIOR.glob("*.json")), PRIOR / "usage.sqlite3", PRIOR / "budget.sqlite3"]
    paths += [path for identifier in CASE_IDS
              for path in sorted((PRIOR / "cases" / identifier).glob("*.json"))]
    controls = dict(prior["controls_sha256"])
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise base.EvaluationError("Prior records must remain regular immutable files.")
        controls[str(path)] = base.file_sha(path)
    return {"kind": "Frozen-input downstream diagnostic; not full pipeline or benchmark",
        "model": base.GPT41_MINI_MODEL, "prompt_version": VERSION,
        "limits": LIMITS, "max_usd": MAX_USD, "request_reservation_usd": 0.03,
        "workers": 1, "publication_authorized": False,
        "bypassed_stages": ["video preparation", "captioning", "indexing", "planner",
                            "tool agent", "retrieval", "inspection", "choice selection"],
        "scoring_policy": "Ground truth is scorer-only; never included in inference input.",
        "controls_sha256": controls,
        "cases": [{"id": identifier, "question": cases[identifier]["question"],
                   "options": cases[identifier]["options"],
                   "reference": cases[identifier]["reference"],
                   "seed_bundle": str(PRIOR / "cases" / identifier / "bundle.json")}
                  for identifier in CASE_IDS]}


def controls_unchanged(manifest):
    return all(Path(path).is_file() and not Path(path).is_symlink()
               and base.file_sha(Path(path)) == digest
               for path, digest in manifest["controls_sha256"].items())


def seed_inputs(case):
    from videomasys.domain.visual_evidence import FrameRef, ObservationRecord
    from videomasys.zerogpu.svp_retrieval import SVPPlan

    saved = read_json(Path(case["seed_bundle"]))
    source = saved["evidence_graph"]
    plan = SVPPlan.model_validate(source["plan"])
    frames = [FrameRef.model_validate(frame) for frame in source["frames"]]
    observations = [ObservationRecord.model_validate(item) for item in source["observations"]]
    frame_ids, needs = {f.id for f in frames}, {need.id for need in plan.evidence_needs}
    if (len(frame_ids) != len(frames) or not frames
        or any(base.file_sha(Path(f.path)) != f.sha256 for f in frames)
        or any(not set(o.frame_ids) <= frame_ids or not set(o.need_ids) <= needs
               for o in observations)
        or case["question"] != saved["question"] or case["options"] != saved["options"]):
        raise base.EvaluationError("Frozen active evidence or question/options do not match.")
    choices = [step for step in saved["trace"] if step["stage"] == "Choice selection"]
    if len(choices) != 1 or choices[0]["status"] != "complete":
        raise base.EvaluationError("Exactly one saved provisional candidate is required.")
    candidate = choices[0]["output"]["selected_choice"]
    if candidate is not None and candidate not in case["options"]:
        raise base.EvaluationError("Saved provisional candidate is not an actual option.")
    retrieval = next(step["output"] for step in saved["trace"] if step["stage"] == "Retrieval")
    graph = {key: deepcopy(source[key]) for key in (
        "video_id", "plan", "frames", "observations", "retrieval_selection",
        "retrieval_coverage_scope", "unresolved",
    )}
    metadata = {"mode": "saved-input diagnostic injection", "not_executed": True,
                "source_bundle": case["seed_bundle"],
                "source_bundle_sha256": base.file_sha(Path(case["seed_bundle"])),
                "source_schema_version": source["schema_version"]}
    # Deliberately exclude historical answers/reviews and video decoding paths.
    bundle = {"key": saved["key"], "question": case["question"], "options": case["options"],
              "selected": deepcopy(saved["selected"]), "trace": [], "replay": metadata}
    return bundle, frames, observations, graph, plan, retrieval, candidate


def load_manifest():
    manifest = read_json(ROOT / "manifest.json")
    if manifest != prepare_manifest() or not controls_unchanged(manifest):
        raise base.EvaluationError("Prepared replay inputs or prior records changed.")
    for case in manifest["cases"]:
        seed_inputs(case)
        if case["options"][case["reference"]["correct_option"]] \
                != case["reference"]["correct_choice"]:
            raise base.EvaluationError("Scoring-only reference does not match the fixed choices.")
    return manifest


def install_seed(runtime, pipeline, bundle, frames, observations, graph, candidate):
    graph["schema_version"] = runtime.SCHEMA_VERSION

    def inspection(_pipeline, current, _question, _plan, _retrieval):
        current["evidence_graph"] = graph
        current["trace"].append({"stage": "Evidence inspection", "status": "seeded",
                                 **bundle["replay"], "attempts": [],
                                 "output": {"frames": graph["frames"],
                                            "observations": graph["observations"]}})
        return frames, observations, graph

    def choice(trace, question, options, images):
        if (question != bundle["question"] or options != bundle["options"]
            or list(images) != [frame.path for frame in frames]):
            raise base.EvaluationError("The saved candidate must use the exact saved evidence.")
        trace.append({"stage": "Choice selection", "status": "seeded", **bundle["replay"],
                      "attempts": [], "output": {"selected_choice": candidate}})
        return candidate

    runtime.inspect_evidence, pipeline.choose_option = inspection, choice


def matching_final_pair(bundle):
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
            or not any(step.get("status") == "failed" for step in later)):
            return False
        expected.update(needs_revision=False, explanation_repair_failed=True,
                        advisories=[*expected["advisories"], RESTORE_NOTE])
    elif later:
        return False
    return expected == bundle.get("verification")


def forbidden(*_args, **_kwargs):
    raise base.EvaluationError("Downstream replay cannot embed, index or inspect new evidence.")


def run_case(case, key, guard, budget, frozen, manifest):
    if (base.sources() != frozen or not controls_unchanged(manifest)
        or read_json(ROOT / "manifest.json") != manifest):
        raise base.EvaluationError("Frozen sources or controls changed before admission.")
    output = ROOT / "cases" / case["id"]
    output.mkdir(mode=0o700)
    bundle, frames, observations, graph, plan, retrieval, candidate = seed_inputs(case)
    base.write_json(output / "started.json",
                    {"started_at": base.utc_now(), **bundle["replay"]}, key)
    base.write_json(output / "seed.json", {"graph": graph, "selected": bundle["selected"],
                                          "retrieval": retrieval, "candidate": candidate}, key)
    runtime = isolated("_injected_replay", "src/videomasys/zerogpu/structured_pipeline.py")
    started, failure = time.monotonic(), None
    try:
        factory = partial(shared.LoggedVLM, output=output, budget=budget, case_id=case["id"])
        app = base.RemotePipeline(provider="openai", token=key, guard=guard, embed=forbidden,
                                  prompt_profile="structured", vlm_factory=factory)
        with app.analysis_session(bundle) as active:
            install_seed(runtime, active, bundle, frames, observations, graph, candidate)
            iterator = runtime.answer_retrieved(active, bundle, case["question"],
                                               case["options"], plan, retrieval)
            try:
                for bundle in iterator:
                    base.atomic_json(output / "checkpoint.json", bundle, key)
            finally:
                iterator.close()
    except Exception as exc:
        failure = base.safe_failure(exc)
    stable = (base.sources() == frozen and controls_unchanged(manifest)
              and read_json(ROOT / "manifest.json") == manifest)
    completed = failure is None and stable and matching_final_pair(bundle)
    eligible = completed and shared.shared.nonabstaining(bundle)
    selected = bundle.get("answer", {}).get("selected_choice")
    report = {"id": case["id"], "completed": completed, "eligible": eligible,
        "correct": eligible and selected == case["reference"]["correct_choice"],
        "saved_candidate": candidate, "selected_choice": selected,
        "candidate_preserved": selected == candidate,
        "abstained": completed and selected is None,
        "retained_reviewed_answer": bundle.get("retained_reviewed_answer", False),
        "review_accepted": bundle.get("verification", {}).get("accepted"),
        "explanation_accepted": bundle.get("verification", {}).get("explanation_accepted"),
        "usage": base.tokens_from_calls(bundle.get("inference", {}).get("calls", [])),
        "elapsed_seconds": time.monotonic() - started, "sources_and_controls_unchanged": stable,
        "error": failure or (None if completed else ["incomplete_final", "No bound final pair."]),
        "decision_transitions": [step["decision_transition"] for step in bundle["trace"]
                                 if "decision_transition" in step]}
    base.write_json(output / "bundle.json", bundle, key)
    base.write_json(output / "raw-trace.json", bundle["trace"], key)
    base.write_json(output / "report.json", report, key)
    return report


def run(manifest):
    if any((ROOT / name).exists() for name in
           ("run-config.json", "usage.sqlite3", "budget.sqlite3", "cases")):
        raise base.EvaluationError("This replay has started; no reruns, resume or ledger resets.")
    frozen = base.sources()
    key = base.read_owner_key()
    config = {"model": base.GPT41_MINI_MODEL, "prompt_version": VERSION, "limits": LIMITS,
              "max_usd": MAX_USD, "request_reservation_usd": 0.03,
              "source_sha256": frozen, "manifest_sha256": base.file_sha(ROOT / "manifest.json"),
              "started_at": base.utc_now(), "explicit_launch_requested": True}
    base.write_json(ROOT / "run-config.json", config, key)
    base.write_json(ROOT / "source-snapshots.json",
                    {name: (REPO / name).read_text() for name in frozen}, key)
    guard = base.UsageGuard(ROOT / "usage.sqlite3", **LIMITS)
    budget = base.Budget(ROOT / "budget.sqlite3", max_usd=MAX_USD)
    (ROOT / "cases").mkdir(mode=0o700)
    rows = []
    with base.selected_model(base.GPT41_MINI_MODEL):
        for case in manifest["cases"]:
            usage = budget.snapshot()
            if usage["budget_exhausted"] or usage["reserved_usd"] or usage["ambiguous_usd"]:
                break
            rows.append(run_case(case, key, guard, budget, frozen, manifest))
    stable = (base.sources() == frozen and controls_unchanged(manifest)
              and base.file_sha(ROOT / "manifest.json") == config["manifest_sha256"])
    summary = {"kind": manifest["kind"], "finished": len(rows) == 2, "results": rows,
        "completed": sum(row["completed"] for row in rows),
        "eligible": sum(row["eligible"] for row in rows),
        "correct": sum(row["correct"] for row in rows),
        "abstentions": sum(row["abstained"] for row in rows), "budget": budget.snapshot(),
        "allowance": guard.snapshot(), "sources_and_controls_unchanged": stable,
        "publication_authorized": False}
    base.write_json(ROOT / "summary.json", summary, key)
    print(json.dumps(summary, indent=2))
    return 0 if stable and summary["completed"] == 2 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if base.PROMPT_VERSION != VERSION:
        raise base.EvaluationError("This downstream replay requires structured-evidence-v6.")
    if args.prepare:
        ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        from replay_rejected_packs import write_exclusive

        write_exclusive(ROOT / "manifest.json", prepare_manifest())
    manifest = load_manifest()
    if not args.run:
        print(json.dumps({"dry_run": True, "cases": list(CASE_IDS), "model": base.GPT41_MINI_MODEL,
                          "limits": LIMITS, "max_usd": MAX_USD, "credential_read": False,
                          "network_calls": 0, "embedding_called": False}, indent=2))
        return 0
    with (ROOT / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(manifest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
