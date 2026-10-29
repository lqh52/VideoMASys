"""Offline replay of 15 captured request-pack rejections; never makes HTTP calls.

Default prints a read-only admission-policy comparison. --write saves it exclusively to a
new artifact. --prepare-smoke fixes nine exposed development cases, but provides
no inference launcher. Earlier ledgers, prompts, outputs and scores stay immutable.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from videomasys.zerogpu.openai import GPT41_MINI_MODEL, OpenAI41MiniVLM  # noqa: E402
from videomasys.zerogpu.remote import MAX_CONTEXT_BOUND, HostedVLMError  # noqa: E402

BASELINE = REPO / "artifacts/benchmark_40_v4_2026-09-08"
LIBRARY = REPO / "artifacts/library_v4_round3_2026-09-08"
ROOT = REPO / "artifacts/payload_replay_2026-09-08"
SMOKE_ROOT = REPO / "artifacts/packing_contract_smoke_2026-09-08"
SMOKE_LIMITS = {"max_runs": 9, "max_requests": 360, "max_requests_per_run": 40,
                "max_usd": 0.50}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_sha(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode())


def frozen_schema_converter(snapshots, config):
    """Execute only the captured pure schema function, never an old module/app."""
    name = "src/videomasys/zerogpu/openai.py"
    source = snapshots[name]
    if digest(source.encode()) != config["source_sha256"][name]:
        raise ValueError("Frozen schema-converter source hash mismatch")
    tree = ast.parse(source)
    selected = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "strict_json_schema")
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "MAX_SCHEMA_BYTES", "_UNSUPPORTED_SCHEMA_KEYS", "OPENAI_SYSTEM_PROMPT",
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    namespace = {"json": json, "HostedVLMError": HostedVLMError, **constants}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), name, "exec"), namespace)
    return namespace["strict_json_schema"], constants["OPENAI_SYSTEM_PROMPT"]


def schema_constraints(schema):
    """Compare validation constraints, resolving local refs and ignoring annotations."""
    def visit(node, depth=0):
        if depth > 40:
            raise ValueError("Unexpected recursive replay schema")
        if "$ref" in node:
            reference = node["$ref"]
            if not reference.startswith("#/"):
                raise ValueError("Only local schema references are supported")
            target = schema
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
            node = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
        result = {}
        for key, value in node.items():
            if key in {"$defs", "definitions", "title", "description", "default", "examples"}:
                continue
            if key == "properties":
                result[key] = {name: visit(child, depth + 1) for name, child in value.items()}
            elif key == "items":
                result[key] = visit(value, depth + 1)
            elif key == "anyOf":
                result[key] = [visit(child, depth + 1) for child in value]
            elif key == "required":
                result[key] = sorted(value)
            else:
                result[key] = value
        return result
    return visit(schema)


def image_hash(path):
    with Image.open(path) as source:
        picture = source.convert("RGB")
    try:
        picture.thumbnail((512, 384))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=90)
        return digest(buffer.getvalue())
    finally:
        picture.close()


def payload_adapter():
    # No constructor/client/credentials: only pure local payload serialization.
    model = object.__new__(OpenAI41MiniVLM)
    model.model, model._image_bounds = GPT41_MINI_MODEL, (512, 384)
    return model


def replay_pack(call, before_bound, legacy_schema, adapter=None, *, system_prompt=None):
    adapter = adapter or payload_adapter()
    before_images = [image_hash(path) for path in call["images"]]
    result = {"before_admitted": False, "before_context_bound": before_bound,
        "before_context_cap": 32768, "prompt_sha256": digest(call["prompt"].encode()),
        "input_schema_sha256": json_sha(call["schema"]),
        "input_image_sha256": [file_sha(path) for path in call["images"]],
        "wire_image_sha256_before": before_images, "image_count": len(before_images),
        "model_called": False, "credential_read": False}
    try:
        wire, bound = adapter._payload(call["prompt"], call["images"],
                                      call["max_output_tokens"], schema=call["schema"])
    except HostedVLMError as exc:
        result.update(after_admitted=False, after_error_code=exc.code)
        return result
    body = json.loads(wire)
    content = body["messages"][1]["content"]
    images = [digest(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))
              for item in content if item["type"] == "image_url"]
    response = body["response_format"]
    prompt_preserved = [item["text"] for item in content if item["type"] == "text"] \
        == [call["prompt"]]
    schema_preserved = schema_constraints(response["json_schema"]["schema"]) \
        == schema_constraints(legacy_schema)
    system_preserved = system_prompt is None or body["messages"][0]["content"] == system_prompt
    limit = getattr(adapter, "_context_limit", lambda _schema: MAX_CONTEXT_BOUND)(call["schema"])
    result.update(after_admitted=True, after_context_bound=bound,
        after_context_cap=limit, admission_policy_changed=limit != 32768, payload_bytes=len(wire),
        wire_sha256=digest(wire), wire_schema_sha256=json_sha(response),
        wire_image_sha256_after=images, images_preserved=images == before_images,
        prompt_preserved=prompt_preserved, schema_constraints_preserved=schema_preserved,
        system_prompt_preserved=system_preserved,
        strict_schema=response["type"] == "json_schema" and response["json_schema"]["strict"])
    if hasattr(adapter, "request_budget"):
        result["numeric_preflight"] = adapter.request_budget(
            call["prompt"], len(call["images"]), call["max_output_tokens"], schema=call["schema"],
        )
    if not (prompt_preserved and system_preserved and schema_preserved and images == before_images
            and result["strict_schema"] and body["model"] == GPT41_MINI_MODEL):
        raise ValueError("Replay changed evidence, order, model, prompt or schema constraints")
    return result


def replay():
    config = json.loads((BASELINE / "run-config.json").read_text())
    snapshots = json.loads((BASELINE / "source-snapshots.json").read_text())
    audit = json.loads((BASELINE / "cost-audit.json").read_text())
    convert, system = frozen_schema_converter(snapshots, config)
    failures = [item for item in audit["failure_classification"]["cases"]
                if item["terminal_category"] == "local_context_guard"]
    if len(failures) != 15 or config["model"] != GPT41_MINI_MODEL:
        raise ValueError("This replay requires the fixed fifteen audited request packs")
    paths = [BASELINE / name for name in (
        "manifest.json", "run-config.json", "source-snapshots.json", "cost-audit.json",
        "usage.sqlite3", "budget.sqlite3", "summary.json",
    )]
    before_controls = {str(path): file_sha(path) for path in paths}
    records = []
    for failure in failures:
        folder = BASELINE / "cases" / failure["id"]
        rejected = []
        for path in sorted(folder.glob("call-*.json")):
            if "-started" in path.name:
                continue
            call = json.loads(path.read_text())
            if call["transport"].get("error_code") == "invalid_request" \
                    and not call["transport"]["attempts"]:
                rejected.append((path, call))
        if len(rejected) != 1:
            raise ValueError("Expected one exact no-HTTP rejection per audited case")
        path, call = rejected[0]
        added = [path, folder / "bundle.json", folder / "report.json"]
        paths += added
        before_controls.update({str(item): file_sha(item) for item in added})
        graph = json.loads((folder / "bundle.json").read_text())["evidence_graph"]
        frames = {frame["path"]: frame for frame in graph["frames"]}
        for image in call["images"]:
            if image not in frames or file_sha(image) != frames[image]["sha256"]:
                raise ValueError("Captured evidence frame no longer matches its original registry")
        converted = convert(call["schema"])
        response = {"type": "json_schema", "json_schema": {
            "name": "video_evidence", "strict": True, "schema": converted,
        }}
        before = (len((system + call["prompt"]).encode())
            + len(json.dumps(response, ensure_ascii=False).encode()) + 256
            + 600 * len(call["images"]) + call["max_output_tokens"])
        if before != failure["conservative_context_bound"] or before <= 32768:
            raise ValueError("Captured request does not reproduce its audited old bound")
        record = replay_pack(call, before, converted, system_prompt=system)
        record.update(id=failure["id"], call_file=str(path), call_sha256=file_sha(path),
            attempt_kind=failure["terminal_schema_attempt"],
            frame_ids_in_image_order=[frames[image]["id"] for image in call["images"]],
            evidence_registry_sha256=json_sha({key: graph[key] for key in (
                "frames", "observations", "unresolved", "plan",
            )}))
        records.append(record)
    controls = {str(path): file_sha(path) for path in paths}
    if any(controls[path] != value for path, value in before_controls.items()):
        raise ValueError("A completed batch control changed during read-only replay")
    return {"kind": "Exact captured-pack admission-policy replay; not packing or accuracy recovery",
        "packs": len(records), "before_rejected": 15,
        "after_admitted": sum(record["after_admitted"] for record in records),
        "network_calls": 0, "credential_read": False,
        "historical_controls_unchanged": True,
        "historical_files_sha256": controls,
        "source_sha256": {str(path.relative_to(REPO)): file_sha(path) for path in
                          [*sorted((REPO / "src/videomasys").rglob("*.py")), Path(__file__)]},
        "records": records}


def smoke_manifest():
    baseline = json.loads((BASELINE / "manifest.json").read_text())
    library = json.loads((LIBRARY / "manifest.json").read_text())
    audit = json.loads((BASELINE / "cost-audit.json").read_text())
    terminal = {item["id"] for item in audit["failure_classification"]["cases"]
                if item["terminal_category"] == "contract_validation"}
    cases = library["cases"] + [case for case in baseline["cases"] if case["id"] in terminal]
    if len(terminal) != 5 or len(cases) != 9 or len({case["id"] for case in cases}) != 9:
        raise ValueError("Development smoke must contain exactly four library plus five failures")
    sources = baseline["label_sources"]
    if any(file_sha(item["path"]) != item["sha256"] for item in sources):
        raise ValueError("Official scoring annotation sources changed")
    return {"kind": "Fixed exposed development regression smoke; NOT held-out accuracy",
        "model": GPT41_MINI_MODEL, "limits": SMOKE_LIMITS, "workers": 2,
        "execution_policy": "At most one cold-index run per case, no automatic retries of "
        "finished cases or additional cohorts. Paid launch requires explicit authorization.",
        "scoring_policy": "References are scoring-only; no labels enter inference. Retained "
        "drafts and abstentions never count as completed nonabstaining answers.",
        "launch_enabled": False, "label_sources": sources,
        "controls_sha256": {str(path): file_sha(path) for path in [
            BASELINE / "manifest.json", BASELINE / "cost-audit.json",
            BASELINE / "usage.sqlite3", BASELINE / "budget.sqlite3",
            LIBRARY / "manifest.json", LIBRARY / "cost-audit.json",
            LIBRARY / "usage.sqlite3", LIBRARY / "budget.sqlite3",
        ]}, "cases": cases}


def write_exclusive(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--prepare-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_smoke:
        value = smoke_manifest()
        write_exclusive(SMOKE_ROOT / "manifest.json", value)
        print(json.dumps({"prepared": 9, "launch_enabled": False, "limits": SMOKE_LIMITS}))
        return 0
    report = replay()
    if args.write:
        write_exclusive(ROOT / "report.json", report)
    accepted = [record for record in report["records"] if record["after_admitted"]]
    print(json.dumps({key: report[key] for key in (
        "kind", "packs", "before_rejected", "after_admitted", "network_calls",
        "credential_read", "historical_controls_unchanged",
    )} | {"after_context_caps": sorted({record["after_context_cap"] for record in accepted}),
        "preserved": all(record["images_preserved"] and record["prompt_preserved"]
                         and record["schema_constraints_preserved"]
                         and record["system_prompt_preserved"] for record in accepted),
        "report_written": args.write}, indent=2))
    return 0 if report["after_admitted"] == 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
