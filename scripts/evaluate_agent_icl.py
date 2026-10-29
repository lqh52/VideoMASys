"""Local-only planner/compiler regression; never retrieves, inspects, or answers.

Default mode tokenizes prompts without loading model weights. --run explicitly
enables real cached-model generation on an otherwise idle local GPU. Saved video
captions are fallible model outputs, not reference answers. Synthetic cases test
structural contracts only, not video accuracy. Python 3.10 uses an evaluation-only
StrEnum compatibility shim; the deployed Space uses a different Python environment.
"""

from __future__ import annotations

import argparse
import ast
import copy
import enum
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULT_SAVED = ROOT / (
    "artifacts/evaluation/holdout_fix_2026-09-07/library/"
    "improved-holdout-fix-v2/20260907T033052.626566Z"
)
CONTEXT_LIMIT = 8192


def import_live():
    if not hasattr(enum, "StrEnum"):

        class StrEnum(str, enum.Enum):  # noqa: UP042 -- Python 3.10 evaluation compatibility
            __str__ = str.__str__
            __format__ = str.__format__

        enum.StrEnum = StrEnum
    sys.path.insert(0, str(ROOT / "src"))
    from videomasys.zerogpu import core, prompts

    return core, prompts


def read_system_prompt(path: Path) -> str:
    """Read the actual production literal without importing spaces or CUDA."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value.strip():
                return value
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        fields = {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        role = fields.get("role")
        if isinstance(role, ast.Constant) and role.value == "system" and "content" in fields:
            value = ast.literal_eval(fields["content"])
            if isinstance(value, str) and value.strip():
                matches.append(value)
    if len(set(matches)) == 1:
        return matches[0]
    raise ValueError(
        "inference.py must contain one unambiguous literal system prompt; no fallback."
    )


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_saved_case(path: Path, name: str, expected_scope: str) -> dict:
    document = json.loads(path.read_text())
    # Deliberate allowlist: never propagate saved answers, verification, or labels.
    source = document["bundle"]
    scenes = [
        {**{key: scene[key] for key in ("index", "start", "end", "caption")}, "frames": []}
        for scene in source["scenes"]
    ]
    if not scenes or len(scenes) > 12:
        raise ValueError(f"Expected 1–12 actual cached scenes in {path}")
    return {
        "id": name,
        "kind": "real_cached_video_captions",
        "source": str(path),
        "source_sha256": file_sha(path),
        "question": document["example"]["question"],
        "options": document["example"]["options"],
        "scenes": scenes,
        "expected_scope": expected_scope,
        "constraints": {"require_relation": "after"} if name == "sofa_mc" else {},
    }


def cases(saved_dir: Path) -> list[dict]:
    sofa = load_saved_case(saved_dir / "01-nextqa-sofa.json", "sofa_mc", "local_event")
    kitchen = load_saved_case(saved_dir / "03-egoschema-kitchen.json", "kitchen_mc", "whole_video")
    kitchen_open = copy.deepcopy(kitchen)
    kitchen_open.update(id="kitchen_open", options=[])

    def fictional(name, question, captions, constraints):
        return {
            "id": name,
            "kind": "synthetic_timeline_contract_only",
            "question": question,
            "options": [],
            "scenes": [
                {
                    "index": index,
                    "start": index * 15.0,
                    "end": (index + 1) * 15.0,
                    "caption": caption,
                    "frames": [],
                }
                for index, caption in enumerate(captions)
            ],
            "expected_scope": "local_event",
            "constraints": constraints,
        }

    return [
        sofa,
        kitchen,
        kitchen_open,
        fictional(
            "novel_independent_attribute",
            "What color is the rim of the bowl on the workbench?",
            [
                "A bowl and measuring jug sit on a workbench.",
                "The bowl remains on the bench while a person moves the jug.",
            ],
            {"independent_only": True, "max_needs": 1},
        ),
        fictional(
            "novel_before_cause",
            "What caused the lab cart to stop at the blue line, considering the moments "
            "immediately before it stopped?",
            [
                "A lab cart rolls along a marked corridor toward a blue line.",
                "A person reaches toward a control beside the corridor; hand contact is unclear.",
                "The cart is stationary at the blue line while the person stands beside it.",
            ],
            {"require_relation": "before"},
        ),
    ]


def question_context(question: str, options: list[str]) -> str:
    context = f"Question: {question}\n"
    context += (
        f"Answer choices (hypotheses): {json.dumps(options)}\n"
        if options
        else "Response mode: open-ended.\n"
    )
    return context


class PromptCaptured(BaseException):
    def __init__(self, prompt, tokens):
        self.prompt, self.tokens = prompt, tokens


def structured_request(core, prompt, schema, tokens):
    """Capture the exact live schema envelope, without returning any completion."""

    def capture(request, images, output_tokens):
        assert not images
        raise PromptCaptured(request, output_tokens)

    pipeline = core.LivePipeline(
        capture, forbidden_embed, "tokenizer-audit", "not-used", prompt_profile="legacy"
    )
    try:
        pipeline.structured([], "Token budget only", prompt, schema, tokens=tokens)
    except PromptCaptured as captured:
        return captured.prompt
    raise AssertionError("Prompt capture unexpectedly generated an output")


def chat_messages(system_prompt, prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def token_count(processor, system_prompt, prompt):
    ids = apply_text_template(
        processor, chat_messages(system_prompt, prompt), tokenize=True, add_generation_prompt=True
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)


def apply_text_template(processor, messages, **kwargs):
    # This text-only harness uses the actual processor template through its
    # tokenizer: older local Transformers processors reject a string system role.
    # No image preprocessing is needed or permitted in these two stages.
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer.apply_chat_template(
            messages, chat_template=processor.chat_template, **kwargs
        )
    return processor.apply_chat_template(messages, **kwargs)  # Unit-test tokenizer fixture.


def pad(pattern, size):
    return (pattern * (size // len(pattern) + 1))[:size]


def budget_audit(core, prompts, processor, system_prompt, plans):
    """Maximum-length fixtures, not a proof over every possible Unicode string."""
    reports = []
    for profile, pattern in (
        ("english", "Visible actions and changing objects. "),
        ("token_dense_ascii", "q9!Z7?x2;B4:"),
    ):
        context = question_context(
            pad(pattern, 2000),
            [str(i) + pad(pattern, 299) for i in range(5)],
        )
        scenes = [
            {"index": i, "start": i * 15, "end": (i + 1) * 15, "caption": pad(pattern, 240)}
            for i in range(12)
        ]
        planner, metadata = prompts.planner_prompt(context)
        stages = [("Planner", planner, core.QAPlan, 600, metadata, None)]
        for name, plan in plans.items():
            compiler_context = core.tool_search_context(context, scenes, plan["scope"])
            compiler, metadata = prompts.tool_prompt(compiler_context, plan)
            stages.append(("Tool agent", compiler, core.ToolProgram, 800, metadata, name))
        for stage, body, schema, output_tokens, metadata, plan_name in stages:
            initial = structured_request(core, body, schema, output_tokens)
            contract = json.dumps(schema.model_json_schema(), separators=(",", ":"))
            # Explicit synthetic token-reservation fixture, not a model response.
            prior = json.dumps({"budget_only_invalid_field": pad(pattern, 3200)})[:3000]
            repair = (
                body
                + "\nRequired schema: "
                + contract
                + "\nYour previous invalid response: "
                + prior
                + "\nValidation error to fix: "
                + pad(pattern, 1000)
                + "\nReturn a complete corrected JSON object. Correct the field types; "
                "do not repeat the invalid structure. Return no explanation."
            )
            for attempt, request in (("initial", initial), ("max_length_repair_fixture", repair)):
                count = token_count(processor, system_prompt, request)
                reports.append(
                    {
                        "profile": profile,
                        "stage": stage,
                        "plan_source": plan_name,
                        "attempt": attempt,
                        "demonstrations": metadata,
                        "input_tokens": count,
                        "reserved_output_tokens": output_tokens,
                        "total_tokens": count + output_tokens,
                        "fits_context": count + output_tokens <= CONTEXT_LIMIT,
                        "prompt": request,
                    }
                )
    return reports


def forbidden_embed(*_args, **_kwargs):
    raise AssertionError("Planner/compiler-only evaluation must never embed or retrieve")


def constraint_issues(case, trace):
    outputs = {entry["stage"]: entry.get("output") for entry in trace}
    plan, program = outputs.get("Planner"), outputs.get("Tool agent")
    if not plan or not program:
        return ["No complete validated planner/compiler pair"]
    errors = []
    if plan["scope"] != case["expected_scope"]:
        errors.append(f"Expected scope {case['expected_scope']}; got {plan['scope']}")
    rules = case.get("constraints", {})
    if rules.get("independent_only") and (
        any(need["depends_on"] for need in plan["evidence_needs"]) or program["relate"]
    ):
        errors.append("Independent attribute task acquired an unnecessary temporal dependency")
    if len(plan["evidence_needs"]) > rules.get("max_needs", 3):
        errors.append("Too many evidence needs for this contract")
    if rules.get("require_relation") and not any(
        call["relation"] == rules["require_relation"] for call in program["relate"]
    ):
        errors.append(f"Expected a {rules['require_relation']} temporal relation")
    if any(call["limit"] > 4 for call in program["locate"] + program["relate"]):
        errors.append("Prompt contract limits retrieval calls to at most four candidates")
    return errors


def run_case(core, case, generate):
    bundle = {"scenes": copy.deepcopy(case["scenes"]), "trace": []}
    pipeline = core.LivePipeline(
        generate, forbidden_embed, f"{MODEL_ID}@{MODEL_REVISION}", "unused",
        prompt_profile="legacy",
    )
    iterator = pipeline.qa(bundle, case["question"], case["options"])
    started = time.monotonic()
    error = None
    try:
        for expected in ("Planner", "Tool agent"):
            yielded = next(iterator)
            if yielded["trace"][-1]["stage"] != expected:
                raise AssertionError("Live QA yield order changed; refusing to advance further")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        iterator.close()  # Do not request a third yield: that would start retrieval.
    issues = constraint_issues(case, bundle["trace"])
    return {
        "case": case,
        "trace": bundle["trace"],
        "error": error,
        "wall_seconds": round(time.monotonic() - started, 3),
        "contract_issues": issues,
        "contracts_passed": error is None and not issues,
        "repairs": sum(max(0, len(stage.get("attempts", [])) - 1) for stage in bundle["trace"]),
        "answer_generated": False,
        "video_accuracy_evaluated": False,
    }


class LocalGenerator:
    def __init__(self, model, processor, torch, system_prompt):
        self.model, self.processor, self.torch = model, processor, torch
        self.system_prompt, self.calls = system_prompt, []

    def __call__(self, prompt, images, tokens):
        if images or len(self.calls) >= 4:
            raise AssertionError(
                "Only two text-only stages with at most one repair each are allowed"
            )
        inputs = apply_text_template(
            self.processor,
            chat_messages(self.system_prompt, prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        input_tokens = int(inputs["input_ids"].shape[-1])
        record = {
            "prompt": prompt,
            "input_tokens": input_tokens,
            "max_new_tokens": tokens,
            "generation_tokens": 0,
        }
        self.calls.append(record)
        if input_tokens + tokens > CONTEXT_LIMIT:
            record["error"] = "Context budget exceeded before generation"
            raise ValueError(record["error"])
        started = time.monotonic()
        outputs = None
        try:
            inputs = inputs.to("cuda")
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    do_sample=False,
                    repetition_penalty=1.08,
                    use_cache=True,
                )
            self.torch.cuda.synchronize()
            tail = outputs[:, input_tokens:]
            record["generation_tokens"] = int(tail.shape[-1])
            raw = self.processor.batch_decode(tail, skip_special_tokens=True)[0]
            record["raw_response"] = raw
            return raw
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["generation_seconds"] = round(time.monotonic() - started, 3)
            del inputs, outputs


def check_idle_gpu(index):
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if int(result.stdout.strip()) > 512:
        raise RuntimeError(f"GPU {index} is not idle; refusing to load weights")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New artifact directory; no overwrite"
    )
    parser.add_argument("--saved-dir", type=Path, default=DEFAULT_SAVED)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument(
        "--run", action="store_true", help="Explicitly load local weights and generate"
    )
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative local GPU index")
    os.environ.update(
        HF_HOME="/vast/s224013995/cache",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
    )
    core, prompts = import_live()
    system_prompt = read_system_prompt(ROOT / "src/videomasys/zerogpu/inference.py")
    selected_cases = cases(args.saved_dir)
    if args.output_dir.exists():
        parser.error("--output-dir must not exist; original artifacts are never overwritten")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
    )
    args.output_dir.mkdir(parents=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 -- Python 3.10
        "local_only": True,
        "mode": "actual_local_generation" if args.run else "tokenizer_only_no_generation",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "gpu_index": args.gpu,
        "context_limit": CONTEXT_LIMIT,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "pydantic")
        },
        "python310_strenum_shim": sys.version_info < (3, 11),
        "system_prompt": system_prompt,
        "source_sha256": {
            name: file_sha(ROOT / name)
            for name in (
                "src/videomasys/zerogpu/core.py",
                "src/videomasys/zerogpu/prompts.py",
                "src/videomasys/zerogpu/inference.py",
                "src/videomasys/agents/tool_sequence.py",
                "scripts/evaluate_agent_icl.py",
            )
        },
        "limitations": "No retrieval, pixel inspection, answers, or video accuracy evaluation. "
        "Local Python/Transformers versions may differ from the hosted Space.",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    # Actual earlier model plans are used only as token-budget inputs, never as completions.
    budget_plans = {}
    for case in selected_cases[:2]:
        saved = json.loads(Path(case["source"]).read_text())
        entry = next(stage for stage in saved["bundle"]["trace"] if stage["stage"] == "Planner")
        budget_plans[case["id"] + "_saved_actual_plan"] = core.QAPlan.model_validate(
            entry["output"]
        ).model_dump()
    budget_plans["mixed_before_after_contract_fixture"] = core.QAPlan.model_validate(
        {
            "scope": "local_event",
            "objective": "Inspect conditions and results around a cart stopping.",
            "evidence_needs": [
                {
                    "id": "cart_stop",
                    "description": "The cart becomes stationary at the marked line.",
                    "temporal_role": "anchor",
                    "depends_on": [],
                },
                {
                    "id": "preceding_condition",
                    "description": "The visible condition before the cart stopped.",
                    "temporal_role": "before",
                    "depends_on": ["cart_stop"],
                },
                {
                    "id": "subsequent_action",
                    "description": "What happens after the cart has stopped.",
                    "temporal_role": "after",
                    "depends_on": ["cart_stop"],
                },
            ],
            "max_intervals": 3,
            "max_inspected_seconds": 60,
            "stop_condition": "The temporal context distinguishes the possibilities.",
            "reasoning": "Dependency order links evidence; it is not chronological order.",
        }
    ).model_dump()
    budgets = budget_audit(core, prompts, processor, system_prompt, budget_plans)
    write_json(args.output_dir / "token_budgets.json", budgets)
    print(
        json.dumps(
            {
                "mode": manifest["mode"],
                "budget_checks": len(budgets),
                "over_context": sum(not row["fits_context"] for row in budgets),
            }
        ),
        flush=True,
    )
    if not args.run:
        return
    check_idle_gpu(args.gpu)
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    torch.set_num_threads(2)
    started = time.monotonic()
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda")
    )  # Explicit placement, no accelerate/device_map.
    torch.cuda.synchronize()
    write_json(
        args.output_dir / "model_load.json",
        {
            "seconds": round(time.monotonic() - started, 3),
            "device": torch.cuda.get_device_name(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        },
    )
    adapter = LocalGenerator(model, processor, torch, system_prompt)
    summaries = []
    generated_plans = {}
    for case in selected_cases:
        adapter.calls = []
        result = run_case(core, case, adapter)
        result["actual_model_calls"] = copy.deepcopy(adapter.calls)
        write_json(args.output_dir / f"{case['id']}.json", result)
        summary = {
            key: result[key] for key in ("error", "wall_seconds", "contracts_passed", "repairs")
        }
        summary["case"] = case["id"]
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        for entry in result["trace"]:
            if entry["stage"] == "Planner" and entry.get("output"):
                generated_plans[case["id"]] = entry["output"]
    write_json(args.output_dir / "summary.json", summaries)
    if generated_plans:
        write_json(
            args.output_dir / "generated_plan_token_budgets.json",
            budget_audit(core, prompts, processor, system_prompt, generated_plans),
        )
    if any(not result["contracts_passed"] for result in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
