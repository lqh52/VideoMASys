# VideoMASys — video understanding & editing prototype

## Hugging Face CPU + hosted VLM demo

The optional Gradio deployment lives in [deploy/zerogpu](deploy/zerogpu/README.md).
It hosts the interface and video pipeline on CPU Basic. The public demo uses
a pinned `gpt-5.4-2026-03-05` directly through OpenAI (`VLM_BACKEND=openai`),
without browser inference or a local GPU allocation. Explicit model selection is restricted to
`gpt-5.6-luna`, `gpt-4.1-mini-2025-04-14`, `gpt-5.4-mini-2026-03-17`, or
the full `gpt-5.4-2026-03-05` snapshot;
reasoning is disabled and there is no automatic model fallback.
The original Qwen3-VL-8B / Hugging Face
provider route remains an explicit alternative. It uses a bounded local semantic cache for the demo;
the PostgreSQL-backed server prototype below remains unchanged.

The Gradio workspace puts one-click library examples above the video upload and
question inputs. **Run analysis** immediately acknowledges the click and locks the
inputs while it runs. Results appear only when available: the answer and timeline,
then expandable pipeline outputs and raw prompts/responses, followed by retrieved
frames and observations. Structured OpenAI stages use strict JSON schemas alongside
the existing semantic checks; this improves format reliability, not a guarantee of
answer accuracy. See the [OpenAI migration](docs/openai_migration_2026-09-07.md) and
the earlier [CPU/API migration and live tests](docs/cpu_hosted_api_2026-09-07.md).

The public Space now explicitly selects `VLM_PROMPT_PROFILE=structured`, version
`structured-evidence-v7`; active role prompts live in
[structured_prompts.py](src/videomasys/zerogpu/structured_prompts.py). It retains
CPU hosting, the four library examples and existing usage limits. This is an
owner-approved demo release, not a claim of passing the historical held-out gates.
See the [deployment and verification record](docs/public_v7_deployment_2026-09-08.md).

## Single-video editor

**Edit** shares the existing video library with **Ask**. Its non-destructive timeline
supports trim handles, splitting, clip reordering, deletion, undo/redo, per-clip mute,
and a title for the first three output seconds. The fixed source strip keeps the
whole video visible: gray footage is excluded from the selected clip, and the
highlighted range is kept. A separate compact sequence row controls clip order.
Sequence playback is local to the
browser; CPU FFmpeg produces 360p rendered previews and 720p MP4 exports.
Manual controls and rendering never call a model or modify the source video.

**Find clips** runs a separate editing agent: an evidence plan → typed cache search
→ visual inspection → a validated cut list. It proposes up to three clips within
a selected maximum of 5–60 seconds. Cuts are sampled-time rough proposals, not
frame-perfect event boundaries; preview and adjust them. Unsupported or invalid
proposals leave the current timeline unchanged. QA prompts and decisions are unaffected.

This first editor is single-source/single-track, not a complete Clipchamp replacement:
no transitions, music library, captions, compositing, or persistent cloud projects.
Download exports before leaving; source, cache, and timelines are temporary.
See [editor architecture and verification](docs/video_editor_2026-09-08.md).

## Development history before v7 publication

The previous public clean role prompts live in [clean_prompts.py](src/videomasys/zerogpu/clean_prompts.py).
They focus on each agent's job and essential evidence rules, without worked examples.
The owner can select the preserved previous prompts with `VLM_PROMPT_PROFILE=legacy`;
`clean` remains the code default; the former public setup used `clean-roles-v2` with GPT-4.1 mini.
Four local live tests compared GPT-4.1 mini and GPT-5.4 mini without reasoning:
both answered the two main questions correctly, but both retained grounding errors.
GPT-5.4 mini did not establish a consistent quality/latency benefit in this small
sample and cost more. See the [actual results](docs/clean_prompt_evaluation_2026-09-07.md).
The trace records the profile/version, and caption caches are
separated by prompt version as well as model.

Two further grounding-prompt candidates were tested on identical frames, then on
a fresh NExT-QA question and an open-ended painting question. They gave mixed
results and were **not promoted** to the public Space. The local experimental
`VLM_PROMPT_PROFILE=tuned` selects `grounded-roles-v2`; `clean` remains the default.
See the [controlled tuning results](docs/prompt_tuning_2026-09-07.md) before using it.

The separate local `svp` profile adds ordered, source-scoped retrieval programs,
paired SVP-style demonstrations, diverse overview retrieval, and bounded
execution-feedback repair. It does not change the public Space or the `clean`
default. See the [implementation and evaluation](docs/svp_logic_2026-09-08.md)
for the actual results and remaining differences from full StructuredVP.
The preserved local hybrid, `svp-roles-v3`, combines these typed/scoped tools
with target-aware temporal ranking and deduplicated evidence selection. See the
[small hybrid comparison](docs/svp_hybrid_2026-09-08.md); it is not deployed.

The new local `VLM_PROMPT_PROFILE=structured` profile keeps that hybrid retrieval
and adds frame-linked observation records, typed claims with explicit dependencies,
and per-claim/option review. The controller resolves citations and renders answers
from those records. **It is experimental, not promoted:** both initial v1 live tests
stopped at answer validation; the corrected `structured-evidence-v2` subsequently
completed both live contract checks. The gift run needed one ID repair and remained
unverified after model review; the kitchen run completed without schema repairs.
Perception accuracy was outside this recheck's scope. See the
[structure, real failures, and corrections](docs/structured_evidence_2026-09-08.md).
The previous local `structured-evidence-v3` simplified all role instructions,
keeps the SVP retrieval examples, and uses `supported`/`unsupported` for choices
and `pass`/`revise` for review. Selection is separate from support; unobserved
alternatives need not be disproved. A subsequent fixed 20+20 live evaluation
completed 18/40 analyses: 6/20 NExT-QA and 7/20 EgoSchema answers were correct,
counting failures and abstentions as incorrect. See the
[40-case benchmark and failure analysis](docs/benchmark_40_2026-09-08.md).
The earlier two-case results remain historical v2 results.
The public `clean` profile remains unchanged.

The previous local `structured-evidence-v4` retained this architecture while adding
request-specific reference schemas, direct option review, complete repair feedback,
and explicit final-answer completion flags. It preserves unused contextual claims
instead of rejecting them, without inventing supporting links. General-purpose
inference and planner/compiler dependency instructions were also clarified.
The release candidate **is not promoted**: the final four-example library run
returned two correct completed answers and stopped on two others. Media, thumbnails,
and official annotation mappings passed independent validation. A separate frozen,
non-overlapping 20+20 evaluation returned 17 non-abstaining final answers and
16 correct answers out of 40, below the 90% completion / 70% accuracy release gates.
The results are documented in the
[v4 release report](docs/structured_v4_release_2026-09-08.md), including all failed
development rounds and the public-release gates. No earlier draft or preliminary
choice is substituted for a failed final answer.

The previous experimental `structured-evidence-v5` fixes the model-specific
structured-request allowance, scoped-context dependency validation, and harmless
reference/option ordering errors, while retaining provenance and support checks.
Raw responses and controller normalization remain visible. See the
[v5 fixes and bounded development checks](docs/structured_v5_fixes_2026-09-08.md).
Its final six-case development recheck completed five analyses (one abstention),
with two correct answers and one remaining tool-dependency failure. The latest
library check was 2/4 correct, so it is **not ready for release**. The public Space
remains unchanged; prior benchmark scores are not rescored.

The earlier local `structured-evidence-v6` separates main-answer review,
explanation-only repair and advisory context notes. Planner coverage is no longer
an answer obligation; choice changes need an explicit reason and explanation-only
repairs cannot change a multiple-choice selection. Evidence references and
positive support paths remain strict. See the
[v6 decision/review changes](docs/structured_v6_decision_review_2026-09-08.md).
A two-case saved-evidence downstream replay completed both cases: the cat answer
was correct, while the sofa still abstained. This is not a new end-to-end score.
This candidate has not been deployed; historical evaluations remain unchanged.

Full GPT-5.4 is available locally with `VLM_BACKEND=openai`,
`VLM_MODEL=gpt-5.4-2026-03-05`, and `VLM_PROMPT_PROFILE=structured`.
It uses `reasoning_effort=none`, `temperature=0`, and the same evidence/schema
limits as the GPT-4.1 mini structured experiment. This is explicit model selection,
not a fallback or a change to the public Space. Each model has a separate video cache.
See the [GPT-5.4 local configuration and evaluation](docs/gpt54_local_2026-09-08.md).
Its four fresh library runs all completed, with three correct answers: sofa, tree,
and kitchen. Painting was wrong despite passing model review. The run cost about
$0.52 and is not a held-out benchmark or a public promotion.

The current local `structured-evidence-v7` fixes object-use scope repair, preserves
complete bounded caption hints for the compiler, and clarifies brief-contact
inspection and equal evidence standards for related choices. A fresh GPT-5.4
library check now answers all four correctly, including painting, for about $0.53.
The cited brush/cup contact was checked visually; cleaning remains an explicit
purpose inference, not an observed pigment-removal claim. Review still has minor
imperfections. See the [painting diagnosis and v7 results](docs/painting_fix_2026-09-08.md).
This is a development check, not held-out release evidence. The user subsequently
approved publishing this version; see the deployment record above.

## Pipeline and deployment

The QA pipeline now validates tool dataflow, inspects retrieved windows more
closely, and separates concise visual choice selection from explanation and
verification. See the [system audit](docs/qa_evaluation_2026-09-07.md) for real
evaluation methods, retained failures, and limitations.
The [hold-out audit](docs/qa_holdout_2026-09-07.md) and
[follow-up fixes](docs/qa_holdout_fix_2026-09-07.md) document open-ended QA failures,
their corrections, and the remaining evidence-quality limitations.

```bash
hf auth login
python scripts/deploy_zerogpu.py YOUR_ACCOUNT/videomasys-demo --dry-run
python scripts/deploy_zerogpu.py YOUR_ACCOUNT/videomasys-demo
```

The uploader creates a private CPU Basic Space and sends only allowlisted source and
deployment files. It does not upload local benchmark videos, artifacts, or secrets,
and will not fall back to paid GPU hardware. Existing hardware is preserved unless
explicitly changed with `--hardware cpu-basic`. Select `VLM_BACKEND=openai` as a
Space variable and add `OPENAI_API_KEY` as a Space secret for OpenAI. Set the
`VLM_MODEL` variable to `gpt-5.4-2026-03-05` and `VLM_PROMPT_PROFILE=structured`
to match the current public demo;
when omitted, the OpenAI backend keeps the initial Luna baseline. For the original
HF backend, leave `VLM_BACKEND=api` and add an inference-only `HF_INFERENCE_TOKEN`
secret. Keys are never sent to the other provider, and no paid fallback is automatic.
The app's initial shared
allowance is five runs / 120 request attempts per ledger; this is not an account
billing cap and its default ephemeral ledger can reset on container replacement.

To also deploy the owner-selected example library, use `--include-library`.
Only files named in `space_library/library.json` are included, without reference
answers. Public visibility also exposes the source, bundled media, and repository
history; review permissions for assets selected for publication. Use
`--allow-public` to explicitly authorize deployment updates to an existing public
Space. That option does not itself change visibility; new Spaces remain private.

Once the Space is running, exercise real inference and cache reuse with a generated
one-second blue clip (this makes billable provider calls, but uploads no benchmark media):

```bash
python scripts/smoke_zerogpu.py YOUR_ACCOUNT/videomasys-demo --repeat
```

The smoke test prints actual outputs and fails if the pipeline does not complete.

For **local UI checks only**, start the explicitly labeled fixture server and run
the browser regression checks in a second terminal. Requires the demo dependencies,
FFmpeg, the local example library, and Playwright with Chromium installed.

```bash
python scripts/preview_zerogpu_ui.py  # terminal 1: http://127.0.0.1:8203
python scripts/smoke_zerogpu_ui.py    # terminal 2: browser checks and screenshots
```

These checks use real video frames with authored **UI FIXTURE** outputs, not model
inference, and make no provider calls. The browser check accepts only a loopback
URL with the fixture badge; it is not a hosted-inference or answer-quality test.

## Separate PostgreSQL server prototype

VideoMASys ingests a video once, builds a hierarchical semantic cache, and answers
questions by retrieving and inspecting only a few relevant intervals. The prototype
uses PostgreSQL + pgvector as its source of truth, Qwen3-VL through a vLLM
OpenAI-compatible API for model-backed roles, and deterministic typed execution.

The current vertical slice deliberately supports **video QA only**. It establishes
extension points for segmentation, tracking, and editing without shipping unfinished
task implementations.

## Runtime pipeline

```text
upload -> ffprobe -> shot candidates -> cached frames -> Qwen3-VL scene analysis
       -> event grouping -> caption embeddings -> PostgreSQL/pgvector

question -> tool-agnostic ReasoningPlan -> typed locate/relate ToolProgram
         -> static validation -> bounded cache execution -> cached-frame inspection
         -> option-by-option evidence judging -> synthesis -> verification -> cited answer
```

The reasoning planner says what evidence is required but cannot name or invoke tools.
A separate tool-sequence agent compiles that plan into typed `locate` and temporal
`relate` calls. A static policy
gate checks dependencies, arguments, and interval/video-second budgets before any tool
runs; invalid output gets one model repair attempt and then a deterministic safe
fallback. The system never executes model-generated Python or shell code. Query-time
visual calls receive only cached frames or bounded motion proxies for retrieved nodes.

## Quick start

Prerequisites: Python 3.11+, FFmpeg/ffprobe, Docker, a Qwen3-VL vLLM endpoint, and a
Qwen3-Embedding endpoint.

```bash
cp .env.example .env
make install
make db-up
make init-db
make api       # terminal 1
make worker    # terminal 2
```

Example Qwen services (adjust model names and GPU allocation for the machine):

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8002 --task embed
```

Upload a video and submit a question:

```bash
curl -F 'file=@sample.mp4' http://localhost:8000/v1/videos
curl -H 'content-type: application/json' \
  -d '{"video_id":"VIDEO_UUID","question":"What happens after the person enters?"}' \
  http://localhost:8000/v1/qa
curl http://localhost:8000/v1/tasks/TASK_UUID
```

OpenAPI documentation is available at `http://localhost:8000/docs`.
The web workspace is available at `http://localhost:8000/`. It supports upload and
indexing status, recent videos, open-ended and multiple-choice QA, live agent-step
progress, source-video playback, retrieved-interval markers, evidence thumbnails,
planner traces, and grounded answer details. The standalone browser mode executes its
model roles locally through WebGPU and exposes the generated artifacts and measured
latencies in an interactive trace.

The browser lab offers four models:

- `LiquidAI/LFM2.5-VL-1.6B-ONNX` is selected by default, with explicit memory
  confirmation before loading. Its official three-graph export uses a dedicated
  ONNX Runtime Web adapter plus a Transformers.js processor constructed from
  the export's nested image configuration and original tokenizer/template.
  FP16 vision/embeddings + Q4 decoder download approximately 2.34 GB (actual
  files, not the model card's smaller estimate). Runtime memory is higher.
  Prefill is chunked to 128 tokens and total context is limited to 4096 tokens.
  `scripts/smoke_lfm_browser.py` is an opt-in real-weight check of the actual JS
  adapter using CPU ONNX sessions; it does not measure browser GPU performance.
- `onnx-community/LFM2.5-VL-450M-ONNX` remains the lower-memory fallback.
- `LiquidAI/LFM2.5-VL-3B-ONNX` as a high-memory quality/latency option.
- `onnx-community/Qwen3.5-4B-ONNX-OPT` as the high-memory quality candidate.

Weights are fetched only after **Load model** or **Run analysis** is clicked and are cached by the
browser. WebGPU requires a secure context. Use `localhost` during development or
serve the application over HTTPS; a plain HTTP LAN address normally cannot expose
`navigator.gpu`.

The repository includes real source videos, extracted scene frames, and benchmark
questions for five NExT-QA cases and one EgoSchema case. They provide media inputs,
not precomputed answers or traces, so this mode works without PostgreSQL or a hosted
VLM server:

```text
http://localhost:8000/
```

The browser prototype is the default, including navigation through the logo. The
older `?browser_demo=1` link still works. Use `?mode=server` explicitly for server
indexing, which requires PostgreSQL and the hosted workers.

After a video and question are selected, **Run live analysis** asks the chosen browser
VLM to generate scene captions, a reasoning plan, typed `locate`/`relate`/`inspect`
calls, bounded evidence observations, an answer, and a verification decision. Scene
captions are cached locally. Retrieval uses normalized MiniLM sentence embeddings on
WASM/CPU, leaving WebGPU memory to the VLM, then applies the generated temporal tool
program and evidence budget. If the
small embedding model cannot load, the trace explicitly records the error and uses a
local TF-IDF vector fallback. Uploaded videos stay in the browser and are sampled into
at most 24 scene windows. Larger VLMs require an explicit warning confirmation.

For plumbing tests without model servers, set `EMBEDDING_PROVIDER=hash`. The hash
embedder is deterministic but has no semantic quality and is not intended for demos
or evaluation. A real VLM endpoint is still required for ingestion and QA.

## API behavior

- `POST /v1/videos`: streams an upload to content-addressed local storage and queues ingestion.
- `GET /v1/videos/{id}`: reports ingestion state and cache statistics.
- `GET /v1/videos`: lists recent videos for the web workspace.
- `GET /v1/videos/{id}/content`: streams an uploaded source video.
- `POST /v1/qa`: queues one typed QA task for an indexed video.
- `GET /v1/tasks`: lists recent tasks, optionally filtered by video.
- `GET /v1/tasks/{id}`: returns state, plan, answer, citations, evidence, and errors.
- `POST /v1/tasks/{id}/cancel`: requests cancellation before the next pipeline step.
- `GET /healthz`: liveness and database connectivity.

Queued jobs remain in PostgreSQL across restarts, and multiple live workers use
`FOR UPDATE SKIP LOCKED` when claiming work. A process crash after a job is claimed
still requires manual requeueing in this slice; expiring leases and retry policy are
the next durability step.

## Storage layout

Raw uploads, representative JPEGs, and generated artifacts live under
`ARTIFACT_ROOT`. Metadata, typed plans, evidence, task traces, and 1024-dimensional
embeddings live in PostgreSQL. The vector dimension is fixed by the initial schema;
changing the embedding model dimension requires a migration and re-index.

## Deliberate prototype limits

- Local filesystem artifacts; use object storage before multi-host deployment.
- At-least-once worker behavior without expiring leases.
- No authentication, tenant isolation, quotas, or high availability yet.
- Scene candidates use FFmpeg visual-change scores; Qwen3-VL adds semantics afterward.
- No audio transcription yet, so spoken-only answers are outside this first slice.
- No arbitrary model-generated code execution.

The fuller architecture is in [docs/system_design.pdf](docs/system_design.pdf).

## Local benchmark smoke tests

The evaluation CLI understands the checked-out NExT-QA and EgoSchema layouts. Resolve
annotations and media without running models using `--dry-run`:

```bash
videomasys-eval --dataset nextqa --root /scratch/DataSets/NExtQA --limit 2 --dry-run
videomasys-eval --dataset egoschema \
  --root /scratch/Projects/3DInfant/SVP_data/EgoSchema --limit 2 --dry-run
```

`--standalone` exercises FFmpeg, the semantic cache, retrieval, and all Qwen3-VL roles
without PostgreSQL. It uses deterministic hash embeddings, so it is a workstation
smoke test rather than a retrieval-quality benchmark:

```bash
videomasys-eval --dataset egoschema \
  --root /scratch/Projects/3DInfant/SVP_data/EgoSchema --limit 1 --standalone \
  --vlm-url http://127.0.0.1:8001/v1 \
  --output artifacts/evaluation/egoschema_results.jsonl
```

Measured single-example results and limitations are recorded in
[docs/benchmark_smoke.md](docs/benchmark_smoke.md).
