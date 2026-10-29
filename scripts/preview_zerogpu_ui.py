"""LOCAL UI FIXTURE ONLY: deterministic display data, never model inference.

Run `.venv/bin/python scripts/preview_zerogpu_ui.py`, then open localhost:8203.
Library videos supply real decoded frames; every generated caption/observation/
answer is explicitly a UI FIXTURE, not a description or analysis of those frames.
Put "error" in the question to exercise a deliberate failure after retrieval.
This script is not included in the deployment allowlist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("GRADIO_TEMP_DIR", f"/tmp/videomasys-ui-fixture-gradio-{os.getuid()}")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from videomasys.zerogpu.core import SceneCache  # noqa: E402
from videomasys.zerogpu.media_sampling import sample_scene_frames  # noqa: E402
from videomasys.zerogpu.ui import STYLE, THEME, build_demo  # noqa: E402

FIXTURE_MODEL = "UI fixture — not model inference"
FIXTURE_SIGNATURE = "local-ui-fixture-v1-never-a-model-cache"


def fixture_step(stage, output):
    """Trace-shape fixture: prompts and responses are explicitly authored test data."""
    return {
        "stage": stage,
        "status": "complete",
        "model": FIXTURE_MODEL,
        "fixture": True,
        "output": output,
        "attempts": [
            {
                "prompt": f"UI FIXTURE: authored {stage} data; no prompt was sent to a model.",
                "raw_response": json.dumps(output),
            }
        ],
    }


def index_fixture(bundle):
    for scene in bundle["scenes"]:
        scene["caption"] = (
            f"UI FIXTURE: placeholder cache caption for scene {scene['index']}. "
            "This text is not an interpretation of the video."
        )
        bundle["trace"].append(
            fixture_step(f"Caption scene {scene['index']}", {"caption": scene["caption"]})
        )
    SceneCache(bundle["database"], FIXTURE_SIGNATURE).save(
        bundle["scenes"], np.eye(len(bundle["scenes"]), dtype=np.float32)
    )
    return bundle


def qa_fixture(bundle, question, options, *, step_delay=0.3):
    scenes = bundle["scenes"]
    positions = sorted({0, len(scenes) // 2, len(scenes) - 1})
    selected = [
        sample_scene_frames(
            Path(bundle["video_path"]), Path(bundle["frame_directory"]), scenes[position], 6
        )
        for position in positions
    ]
    scene_ids = [scene["index"] for scene in selected]
    plan = {
        "objective": "UI FIXTURE: demonstrate how a multi-stage result is displayed.",
        "evidence_needs": [
            {
                "id": "anchor",
                "description": "UI FIXTURE: example initial-event evidence need.",
                "temporal_role": "anchor",
                "depends_on": [],
            },
            {
                "id": "outcome",
                "description": "UI FIXTURE: example later-event evidence need.",
                "temporal_role": "outcome",
                "depends_on": ["anchor"],
            },
        ],
        "temporal_constraints": ["UI FIXTURE: the illustrative outcome follows the anchor."],
        "option_strategy": "UI FIXTURE: no answer choices are evaluated.",
        "stop_condition": "UI FIXTURE: stop after displaying the authored evidence cards.",
        "reasoning": "UI FIXTURE: this plan exists only to test the interface layout.",
        "max_intervals": 3,
        "max_inspected_seconds": 60,
    }
    program = {
        "version": "2.0",
        "compiler": "model",
        "rationale": "UI FIXTURE: authored typed-program display; not model-generated.",
        "max_intervals": 3,
        "max_inspected_seconds": 60,
        "locate": [
            {
                "id": "l1",
                "query": "UI FIXTURE: initial event",
                "limit": 2,
                "evidence_need_ids": ["anchor"],
            },
            {
                "id": "l2",
                "query": "UI FIXTURE: later event",
                "limit": 2,
                "evidence_need_ids": ["outcome"],
            },
        ],
        "relate": [
            {
                "id": "r1",
                "source": "l1",
                "relation": "after",
                "horizon_seconds": 180,
                "limit": 2,
                "evidence_need_ids": ["outcome"],
            }
        ],
        "inspect": {"sources": ["l1", "l2", "r1"], "evidence_need_ids": ["anchor", "outcome"]},
    }
    for stage, output in (("Planner", plan), ("Tool agent", program)):
        time.sleep(step_delay)
        bundle["trace"].append(fixture_step(stage, output))
        yield bundle

    # These fixed scores intentionally test rendering, not semantic retrieval.
    retrieval = {
        "fixture_note": "UI FIXTURE: manually selected spans and illustrative scores.",
        "selected": selected,
        "calls": [
            {"call": program["locate"][0], "hits": [{"scene": scene_ids[0], "score": 0.82}]},
            {"call": program["locate"][1], "hits": [{"scene": scene_ids[-1], "score": 0.76}]},
            {
                "call": program["relate"][0],
                "hits": [{"scene": index, "score": 0.5} for index in scene_ids[1:]],
            },
            {
                "call": "inspect",
                "selected": scene_ids,
                "inspected_seconds": sum(scene["end"] - scene["start"] for scene in selected),
                "budget_seconds": 60,
            },
        ],
    }
    time.sleep(step_delay)
    bundle["selected"] = selected
    bundle["trace"].append(fixture_step("Retrieval", retrieval))
    yield bundle
    time.sleep(step_delay)
    if "error" in question.lower():
        raise ValueError("UI FIXTURE: intentional test error after retrieval; no model ran.")
    inspection = {
        "evidence": [
            {
                "scene": scene["index"],
                "observation": (
                    f"UI FIXTURE: observation card for scene {scene['index']}. "
                    "The displayed frames are real, but this statement is authored test data."
                ),
                "relevant": True,
                "uncertainty": "UI FIXTURE: this is a display test, not visual interpretation.",
            }
            for scene in selected
        ]
    }
    bundle["trace"].append(fixture_step("Evidence inspection", inspection))
    yield bundle
    time.sleep(step_delay)
    answer = {
        "answer": (
            "UI FIXTURE — this is an authored interface example, not a video answer. "
            "The selected windows, frame gallery, observations, citations, and trace panels "
            "are shown together to check the layout. No model inference took place."
        ),
        "selected_option": 0 if options else None,
        "selected_choice": options[0] if options else None,
        "inference": "UI FIXTURE: reasoning bridge shown here is authored layout test data.",
        "supporting_scenes": scene_ids,
        "limitations": [
            "UI FIXTURE: selected choice and citations are test data, not conclusions."
        ],
    }
    bundle["answer"] = answer
    bundle["trace"].append(fixture_step("Answer", answer))
    yield bundle
    time.sleep(step_delay)
    verdict = {"accepted": True, "issues": ["UI FIXTURE: simulated review state; no model ran."]}
    bundle["verification"] = verdict
    bundle["status"] = "verified"
    bundle["trace"].append(fixture_step("Verifier", verdict))
    yield bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8203)
    parser.add_argument("--step-delay", type=float, default=0.3)
    parser.add_argument(
        "--library-root", type=Path, default=Path(__file__).resolve().parents[1] / "space_library"
    )
    args = parser.parse_args()
    if not 0 <= args.step_delay <= 10:
        parser.error("--step-delay must be between 0 and 10 seconds.")
    # build_demo invokes real prepare_video/FFmpeg, but this cache is never reused
    # by the production pipeline and is removed when the fixture server exits.
    with tempfile.TemporaryDirectory(prefix="videomasys-ui-fixture-") as cache:
        pipeline = SimpleNamespace(model_id=FIXTURE_MODEL, signature=FIXTURE_SIGNATURE)
        demo = build_demo(
            pipeline,
            index_fixture,
            partial(qa_fixture, step_delay=args.step_delay),
            cache_root=cache,
            library_root=args.library_root,
        )
        print("LOCAL UI FIXTURE ONLY — authored outputs, no GPU or model inference.", flush=True)
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            css=STYLE,
            theme=THEME,
            allowed_paths=[str(args.library_root.resolve()), cache],
            max_file_size="100mb",
            show_error=True,
            share=False,
        )


if __name__ == "__main__":
    main()
