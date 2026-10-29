# VideoMASys

VideoMASys is a research prototype for understanding and editing video. It can answer
open-ended or multiple-choice questions, show the evidence used for an answer, and
create editable rough cuts from natural-language instructions.

The repository includes:

- a browser-first video QA workspace that runs visual-language models with WebGPU;
- a FastAPI, PostgreSQL, pgvector, and worker-based server pipeline;
- a CPU-hosted Gradio application that can call OpenAI or Hugging Face inference;
- a non-destructive single-video editor with AI-assisted clip selection;
- evaluation scripts, tests, and small example videos.

> [!IMPORTANT]
> This is a research prototype, not a production service. Do not upload private or
> sensitive media without reviewing where frames, prompts, and generated data are sent.

## Quick start: browser mode

Browser mode is the easiest way to explore the project. PostgreSQL serves application
state, while the selected visual-language model and retrieval pipeline run in the
browser. A separate model server and background worker are not required for this mode.

### Requirements

- Python 3.11 or newer
- Docker with Compose
- FFmpeg and ffprobe
- a WebGPU-capable browser for live browser inference

Install FFmpeg with your operating system's package manager, then clone and start the
application:

```bash
git clone https://github.com/lqh52/VideoMASys.git
cd VideoMASys
cp .env.example .env
make install
make db-up
make init-db
make api
```

Open <http://localhost:8000>. Select a bundled example or upload a video, choose a
browser model, and click **Run live analysis**. Model weights are downloaded only after
you load a model or begin an analysis and are then cached by the browser.

The default browser model is approximately 2.34 GB. If the browser cannot provide
WebGPU, try the smaller model from the model selector. WebGPU normally requires
`localhost` or HTTPS.

Stop the database when finished:

```bash
make db-down
```

## Choose a runtime

VideoMASys supports three related runtimes:

| Runtime | Best for | Model execution | Persistent database |
| --- | --- | --- | --- |
| Browser mode | Trying the project locally | WebGPU in the browser | PostgreSQL |
| Server mode | API development and worker experiments | OpenAI-compatible VLM endpoints | PostgreSQL + pgvector |
| Gradio deployment | Sharing a bounded demo | OpenAI or Hugging Face hosted inference | Local SQLite cache |

The browser UI is the default at <http://localhost:8000>. Add `?mode=server` to the URL
to use the server-backed ingestion and QA workflow.

## Server mode

Server mode ingests each video once, stores a semantic cache in PostgreSQL, and answers
questions by retrieving and inspecting a bounded set of relevant intervals.

### 1. Start the application dependencies

Follow the browser quick start through `make init-db`. Configure `.env` if your model
endpoints differ from the defaults.

### 2. Start model endpoints

The following example uses vLLM. GPU requirements depend on the selected models and
are not installed by this project.

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8002 --task embed
```

For plumbing tests, set `EMBEDDING_PROVIDER=hash` in `.env`. The hash embedder is
deterministic but has no semantic quality and should not be used for evaluation.

### 3. Start the API and worker

Run these commands in separate terminals:

```bash
make api
```

```bash
make worker
```

Open <http://localhost:8000/?mode=server>, or use the API directly.

## API example

Upload a video:

```bash
curl -F 'file=@sample.mp4' http://localhost:8000/v1/videos
```

Copy the returned video ID. Wait until `GET /v1/videos/{id}` reports `ready`, then
submit a question:

```bash
curl -H 'content-type: application/json' \
  -d '{"video_id":"VIDEO_UUID","question":"What happens after the person enters?"}' \
  http://localhost:8000/v1/qa
```

Poll the returned task ID for the answer, evidence, and execution trace:

```bash
curl http://localhost:8000/v1/tasks/TASK_UUID
```

Interactive API documentation is available at <http://localhost:8000/docs>.

### Main endpoints

- `POST /v1/videos` uploads a video and queues ingestion.
- `GET /v1/videos/{id}` reports ingestion state and cache statistics.
- `GET /v1/videos/{id}/content` streams the uploaded source video.
- `POST /v1/qa` queues a QA task for an indexed video.
- `GET /v1/tasks/{id}` returns the plan, answer, citations, evidence, and errors.
- `POST /v1/tasks/{id}/cancel` requests cancellation.
- `GET /healthz` reports service and database health.

## Video editor

Switch from **Ask** to **Edit** in the Gradio workspace. The editor supports:

- trimming, splitting, reordering, and deleting clips;
- undo and redo;
- per-clip mute controls;
- a title over the first three output seconds;
- local sequence playback;
- 360p preview rendering and 720p MP4 export;
- AI-assisted **Find clips** rough-cut proposals.

Manual editing and rendering do not call a model or change the source video. **Find
clips** uses the semantic cache, typed retrieval, and fresh visual inspection. Suggested
cuts are approximate sampled-time boundaries, so preview and adjust them before export.

The editor is currently single-source and single-track. It does not provide transitions,
captions, a music library, compositing, or persistent cloud projects. See the
[editor design](docs/video_editor_2026-09-08.md) for implementation details.

## How video QA works

```text
upload -> ffprobe -> scene candidates -> cached frames -> scene analysis
       -> event grouping -> caption embeddings -> PostgreSQL/pgvector

question -> evidence plan -> typed locate/relate program -> static validation
         -> bounded retrieval -> visual inspection -> choice judging
         -> answer composition -> verification -> cited answer
```

The reasoning planner describes the evidence it needs but cannot execute tools. A
separate compiler produces typed retrieval operations. Before execution, a static gate
checks tool dependencies, arguments, and time budgets. Invalid model output gets one
bounded repair attempt followed by a deterministic safe fallback. The application never
executes model-generated Python or shell code.

## Configuration

Copy `.env.example` to `.env` and change only the values needed for your environment.
The most commonly used settings are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL connection used by the API and worker | Compose database on port 5432 |
| `ARTIFACT_ROOT` | Uploads, frames, and generated files | `./artifacts` |
| `VLM_BASE_URL` | OpenAI-compatible visual-language endpoint | port 8001 |
| `VLM_MODEL` | Model name sent to the VLM endpoint | `Qwen/Qwen3-VL-8B-Instruct` |
| `EMBEDDING_PROVIDER` | `openai` or deterministic `hash` | `openai` |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embedding endpoint | port 8002 |
| `EMBEDDING_MODEL` | Embedding model name | `Qwen/Qwen3-Embedding-0.6B` |
| `MAX_UPLOAD_BYTES` | Maximum server upload size | 20 GiB |

Keep credentials in `.env` or your deployment platform's secret store. `.env` and
other environment variants are excluded from Git; `.env.example` contains only safe
defaults.

## Hugging Face Spaces deployment

The optional Gradio deployment runs sampling, caching, and orchestration on CPU and
uses a hosted inference provider for VLM calls. It does not require a local GPU.

Install the deployment client, authenticate, and perform a dry run first:

```bash
.venv/bin/pip install huggingface_hub Pillow
.venv/bin/hf auth login
.venv/bin/python scripts/deploy_zerogpu.py ACCOUNT_NAME/videomasys-demo --dry-run
.venv/bin/python scripts/deploy_zerogpu.py ACCOUNT_NAME/videomasys-demo
```

Add `--include-library` to publish the bundled examples. The uploader sends only an
allowlist of application files and never uploads `.env`, generated artifacts, benchmark
outputs, or reference answers.

For an OpenAI-backed Space, configure these in the Space settings:

- variable: `VLM_BACKEND=openai`
- variable: `VLM_MODEL` set to a model supported by the deployment
- variable: `VLM_PROMPT_PROFILE=structured`
- secret: `OPENAI_API_KEY`

For the Hugging Face inference route, use `VLM_BACKEND=api` and store
`HF_INFERENCE_TOKEN` as a Space secret. Provider selection is explicit; the application
does not silently switch providers or models.

Read [the complete deployment guide](deploy/zerogpu/README.md) before making a Space
public or changing its hardware.

## Evaluation

The evaluation CLI expects the standard directory layouts for NExT-QA or EgoSchema.
Use repository-relative dataset directories or pass a path appropriate for your own
environment.

Validate annotations and media without calling a model:

```bash
.venv/bin/videomasys-eval \
  --dataset nextqa \
  --root ./datasets/nextqa \
  --limit 2 \
  --dry-run

.venv/bin/videomasys-eval \
  --dataset egoschema \
  --root ./datasets/egoschema \
  --limit 2 \
  --dry-run
```

Run the standalone media, cache, retrieval, and VLM path without PostgreSQL:

```bash
.venv/bin/videomasys-eval \
  --dataset egoschema \
  --root ./datasets/egoschema \
  --limit 1 \
  --standalone \
  --vlm-url http://localhost:8001/v1 \
  --output ./artifacts/evaluation/egoschema-results.jsonl
```

The standalone command uses deterministic hash embeddings. It is a workflow smoke test,
not a retrieval-quality benchmark. See [benchmark notes](docs/benchmark_smoke.md) for
measured examples and limitations.

## Development

Run lint and tests from the repository root:

```bash
make lint
make test
```

Some evaluation tests intentionally pin historical prompt versions and preserved local
artifacts. A version-guard failure means the current prompt profile differs from the
archived experiment; do not bypass the guard when reporting historical results.

Useful local UI checks:

```bash
.venv/bin/pip install -r deploy/zerogpu/requirements.txt
.venv/bin/python scripts/preview_zerogpu_ui.py
.venv/bin/python scripts/smoke_zerogpu_ui.py
```

The preview uses authored fixture outputs and makes no provider calls. It is a UI test,
not an answer-quality evaluation.

### Repository layout

```text
deploy/              Hugging Face Spaces entrypoint and deployment metadata
docs/                architecture, evaluation, and release notes
migrations/          PostgreSQL schema
scripts/             deployment, smoke-test, and evaluation utilities
space_library/       small example videos and thumbnails
src/videomasys/      application source
tests/               unit, integration, UI, and archival evaluation tests
```

## Current research profile

The hosted configuration uses the `structured` prompt profile,
`structured-evidence-v7`. It combines typed, source-scoped retrieval with frame-linked
observations, explicit claim dependencies, choice review, and answer verification.
Strict schemas improve output reliability but do not guarantee visual accuracy.

Historical experiments and their limitations remain available in `docs/`:

- [QA system audit](docs/qa_evaluation_2026-09-07.md)
- [hold-out evaluation](docs/qa_holdout_2026-09-07.md)
- [structured evidence design](docs/structured_evidence_2026-09-08.md)
- [v4 release evaluation](docs/structured_v4_release_2026-09-08.md)
- [v7 painting diagnosis](docs/painting_fix_2026-09-08.md)
- [public deployment record](docs/public_v7_deployment_2026-09-08.md)

These reports are historical evidence, not claims that the prototype meets production
accuracy or reliability requirements.

## Limitations

- Uploaded artifacts are stored on the local filesystem in server mode.
- Worker recovery does not yet use expiring leases.
- The server prototype has no authentication, tenant isolation, or high availability.
- Scene candidates use FFmpeg visual-change scores and may miss short actions.
- Audio is not transcribed, so spoken-only answers are outside the current scope.
- Browser and hosted models may be large, slow, rate-limited, or unavailable.
- Generated answers and rough cuts must be reviewed by a person.

For the full architecture, see [the system design](docs/system_design.pdf).
