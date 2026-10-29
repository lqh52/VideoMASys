---
title: VideoMASys
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: 3.12.12
app_file: app.py
suggested_hardware: cpu-basic
short_description: Understand video and create editable AI-assisted rough cuts
---

# VideoMASys — understand & edit video

Upload a short video, ask a question, and inspect the actual plan, typed tool calls,
retrieval scores, selected frames, answer, and verifier output. No browser GPU or
locally downloaded model is required. No prerecorded answers are used.

This is a **demo**, not an always-on production service. The default deployment
hosts Gradio, video sampling, the cache and agent orchestration on CPU Basic.
The public demo selects `VLM_BACKEND=openai`: visual-language calls use
the pinned `gpt-5.4-2026-03-05` model with reasoning disabled and strict JSON
schemas for structured agent stages. The Space explicitly selects
`VLM_PROMPT_PROFILE=structured` (`structured-evidence-v7`). Allowed alternatives
remain `gpt-5.6-luna`, `gpt-4.1-mini-2025-04-14`, and
`gpt-5.4-mini-2026-03-17`. The original `api` backend remains available for
`Qwen/Qwen3-VL-8B-Instruct:featherless-ai` through Hugging Face's inference router.
There is no local VLM download, CUDA initialization or GPU allocation in either mode.
The interface can start without an inference token or a loaded embedding model;
live analysis still requires a working provider, valid credentials and available quota.
Moving the frontend off a GPU Space avoids coupling page startup to that Space's
GPU scheduler. It does not guarantee the remote provider's availability or latency.

CPU MiniLM embeddings use [FastEmbed](https://qdrant.github.io/fastembed/) and ONNX
Runtime, initialized lazily when needed. They support a small SQLite/exact-cosine
video cache; the full server architecture's PostgreSQL/pgvector path remains
separate. The app does not run a persistent vLLM daemon. Hosted model weights and
serving implementations are provider-controlled, not pinned to our local weight
revision. The existing planner, typed tools, evidence bounds and inspectable traces
remain part of the application.

## Using the demo

The example cards include small, pre-generated thumbnails taken from the videos.
Click the image or its title to select an example. Browsing does not run inference.

- Choose one of the demo's four example questions at the top, or upload a
  video in the video/question workspace. Examples load the video, question, and
  answer choices with one click; they do not run inference automatically.
- Click **Run analysis**. The button immediately changes to **Running analysis…**,
  and the inputs lock until the run finishes or fails. Status messages identify
  the current stage without inventing a completion percentage.
- Results stay hidden before a run produces data. The answer keeps its original
  video/question context, citations, and model-review status. A timeline shows the
  actual retrieved spans; the evidence section separates sampled frames and fresh
  observations from expandable cached captions. Highlighted spans are not a claim
  that every frame in those windows was inspected.
- Below the answer and above the evidence, expand **Inspect the pipeline** for
  actual planner needs, typed tool calls,
  retrieval scores, and model outputs. **Raw prompts & responses** retains the
  complete trace, including validation repairs and failures.
- Open/closed step choices persist while results stream in; new runs reset them.
  **Expand all** / **Collapse all** controls also support keyboard activation.
  Step outputs grow with their content so the page scrolls normally, without
  separate inner scrollbars for tool programs or answers.
- Embedded-page height updates follow streamed results and native step/caption
  expansion, so the Hugging Face page can scroll to all evidence and the footer.
  Width changes also refresh wrapping; iframe-height responses do not restart
  the resize cycle.

## Bounds and limitations

### Video editing

Switch to **Edit** to use the same library video or upload another. The source strip
always shows the full video at a fixed scale. Drag its trim handles to set the kept
range; excluded footage stays visible in gray, including after you release a handle.
Both ends fit in view. The compact **Sequence** row selects and reorders clips.
You can also set source in/out times, split at the playhead,
delete, undo/redo, mute individual clips, or add a title for the first three output
seconds. Reset restores the full source and is undoable. Browser playback previews
the selected sequence without model calls. **Render preview** makes a 360p file;
**Export MP4** makes a 720p H.264/AAC file. These CPU operations preserve the source
and its audio unless a clip is muted. Download exports before leaving.

**Find clips** creates a new rough cut from an instruction such as "Keep the moments
when the boy unwraps his present." Its separate agent uses the existing cache,
typed retrieval and fresh visual inspection, not the QA answer/reviewer. Actual
tool calls, evidence and repairs appear under **How AI chose the clips**. Up to
three supported clips must fit the selected 5–60 second maximum; this is not a
promise to fill the entire requested duration. Sampled seek times are approximate
cut suggestions, so preview and adjust them. Failed proposals do not replace your
timeline. Find clips uses the same hosted-model quota as QA; manual edits and
rendering do not use model tokens.

This is a single-video, single-track editor: no transitions, captions, music library,
compositing or saved cloud projects. Editable sequences allow up to 12 clips/180s;
exports require at least one clip. Render jobs have a four-minute timeout,
cancellation, CPU/thread bounds, and a 512 MiB/32-file output cache. A full cache
stops new renders explicitly; it does not silently delete another user's export.
Files and session state are temporary and may disappear on restart.

### Video understanding

- Uploads: at most 100 MiB, 180 seconds, 4K; one analysis at a time, four queued.
- Index: uniform temporal windows, four endpoint-aware frames per window. These are **not**
  model-predicted scene boundaries; fast actions between samples may be missed.
- QA: at most three intervals spanning 60 seconds, six fresh samples per selected
  window (18 images per visual call), with bounded text and output lengths. Sample times are approximate
  FFmpeg seek targets, not measured frame presentation timestamps.
- The planner seeks discriminating actions, then a separate model compiles typed,
  source-scoped `locate` / `relate` / `overview` / `inspect` calls. Static validation rejects unused retrieval
  outputs and disconnected dependencies. Every repair remains in the trace.
- The public `structured` profile uses fictional SVP-style Question → Plan → Program
  examples and frame-linked observations, observed/inferred claims, and explicit
  conclusion dependencies. The planner sees only the question/choices; the compiler
  receives fallible, bounded cached captions. Actual prompts and their profile/version
  are recorded in the trace. The former public `clean-roles-v2` / GPT-4.1 mini
  configuration remains available explicitly; `clean` is still the code default.
  Four fresh local GPT-5.4/v7 library runs completed correctly, but these are exposed
  development examples, not held-out accuracy evidence. Perception and review errors
  remain possible; structured links are not proof that a visual claim is true.
- Whole-video plans use independent evidence needs and representative overview
  retrieval, without forcing a beginning/middle/end chain. Local-event plans seek
  the relevant action chain and can search repeated object uses anywhere in the video.
  The compiler has one bounded execution-feedback repair if retrieval coverage fails.
  Cached captions are never a substitute
  for fresh visual inspection; temporal coverage does not mean every event was seen.
- Temporal search includes the anchor window, since a coarse window can contain
  multiple actions. `anchor_overlap` is not proof of event order. Eligible windows
  are reranked using the target need's caption-query cosine; temporal proximity is
  a separate tie-break. The bounded selector prioritizes evidence-need coverage.
- Inspection uses one model call per selected window to reduce image/scene mixing.
  Answering and verification see the same selected images with explicit mappings.
  Scene citations are constrained to the actual IDs. The verifier checks visible
  support while allowing grounded purpose inference. It uses the same underlying
  model, so its agreement is not an independent guarantee of accuracy.
- For a brief/unclear key interaction, the inspector may request one closer look:
  12 samples from a clipped ±2-second region inside an already selected window.
  Its final six images retain the original window's first and last samples plus
  four focused samples, ordered chronologically. Answering and verification still
  receive at most 18 images. The request, span, retained context times, and actual
  model call remain in the trace. This does not prove that a subjective purpose
  was directly visible.
- Multiple-choice QA first uses a concise visual choice-selection call, separate
  from explanation formatting. The Answer agent builds linked claims and compares
  every option as supported/unsupported; a different selection requires an explicit
  evidence-based reason. Unsupported does not mean disproved. The controller maps exact choice text to a zero-based
  index and records the transformation, with no fuzzy/default-answer fallback.
- Open-ended QA skips choice selection and uses the same linked evidence/claim
  representation with no option comparisons. It gives a substantive conclusion
  or a specific evidence limitation, not a bare placeholder. It never requires
  answer choices. Both modes retain the same frame and retrieval budgets.
- Tool validation reports missing evidence IDs, their descriptions, and how to
  connect their locate outputs to inspection. A failed repair remains a visible
  failure; no retrieval result or completed tool program is fabricated.
- Review separates substantive answer issues, explanation edits and advisory
  comments. One bounded correction and final review may use the exact same images;
  explanation-only repair cannot change the choice. An optional polish failure can
  retain only the exact already-reviewed answer, with its explanation limitation.
  Original drafts, feedback and final pair remain in the trace. There is no
  accept-until-pass loop.
- Structured repair retains the complete prior response and concrete validation
  feedback; failed outputs remain in the trace. Hosted
  generation uses the provider's supported settings, not a guarantee of bit-for-bit
  agreement with the optional local PyTorch backend.
- One model repair attempt per malformed stage, then a visible failure. No invented
  plan, retrieval hits, observations, or answer is substituted.
- Answers rejected by the verifier stay visibly unverified. Verification is also
  model-generated, not a guarantee of correctness.
- Hosted API rate limits, provider errors and request timeouts can prevent a live
  run. A successful model response is not guaranteed by CPU hosting. The app keeps
  its single-analysis queue and frame/output limits; it does not silently substitute
  a recorded answer or automatically switch to another paid provider.
- Captions and vectors are reused by video hash and configuration signature.
  Model and prompt versions have separate cache identities, so GPT-5.4/v7 does
  not silently reuse old model captions. The
  default cache is ephemeral: it can disappear on a Space restart. Configure
  `DEMO_CACHE_DIR` on durable storage for persistence. Do not publish private uploads.
- Gradio removes temporary uploaded media after its retention window. Do not treat
  this Space as permanent storage. No upload is committed to a Hub repository.
  Use non-sensitive videos in a public demo: generated media URLs are not
  per-user-private, and the shared scene cache is separate from Gradio's upload
  retention window. Questions, sampled frames and the text needed by each agent
  are sent to the hosted inference service; do not submit private or confidential
  media without reviewing the provider's data-handling terms.

## Deploy

The owner explicitly chooses hardware. `suggested_hardware` is metadata and does
not allocate hardware. New deployments default to CPU Basic (`cpu-basic`);
code-only updates preserve an existing Space's hardware and sleep timeout. To
explicitly migrate an owner-approved public demo to CPU Basic:

```bash
python scripts/deploy_zerogpu.py lqh52/videomasys-demo --include-library --allow-public --hardware cpu-basic
```

CPU Basic requires no `--sleep-seconds`; that flag is rejected for CPU hardware.
The script never changes visibility, configures secrets, enables auto-recharge,
or starts additional replicas. Confirm the returned runtime and perform an explicit
live API smoke test before calling the deployment ready. A code-only update to an
existing L4 Space does **not** turn off its paid GPU; hardware migration must be
explicit.

### API configuration and usage control

- For direct OpenAI, set the Space **variable** `VLM_BACKEND=openai` and add
  `OPENAI_API_KEY` as a Space **secret**. Set `VLM_MODEL` explicitly to
  `gpt-4.1-mini-2025-04-14`, `gpt-5.4-mini-2026-03-17`, `gpt-5.6-luna`, or
  the full `gpt-5.4-2026-03-05` snapshot (current public selection).
  Omitting `VLM_MODEL` keeps Luna as the initial low-cost baseline. The key stays on the
  server and is sent only to `api.openai.com`, never to Hugging Face's inference
  router. Use a project-scoped key, and configure billing in your OpenAI account.
  Hugging Face credit does not pay for OpenAI requests.
- Luna uses Chat Completions, explicit `reasoning_effort=none`, standard service
  tier, `store=false`, bounded `max_completion_tokens`, and resized image inputs
  with explicit detail. Structured stages use strict JSON Schema; the separate
  exact-choice step remains plain text. Core evidence/dataflow validation and
  bounded semantic repair remain in place. Strict JSON is not proof of correctness.
- GPT-4.1 mini uses the same bounded Chat Completions/schema path but omits the
  unsupported reasoning parameter and uses `temperature=0`. Its API snapshot is
  pinned; the serving implementation remains provider-managed. Its standard rates
  are $0.40 / million input, $0.10 / million cached input, and $1.60 / million output
  tokens. See [official Mini pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
- GPT-5.4 mini uses the pinned March 17 snapshot, explicit `reasoning_effort=none`,
  and the same image/output/request limits. Standard rates are $0.75 / million input,
  $0.075 / million cached input, and $4.50 / million output tokens. It is more
  expensive per token than GPT-4.1 mini; selecting it never increases the demo's
  request allowance. See [official 5.4 Mini pricing](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
- Full GPT-5.4 uses the March 5 snapshot, `reasoning_effort=none` and `temperature=0`.
  It preserves the evidence, image, output and retry limits; structured requests
  have the same 65,536-token app allowance as the GPT-4.1 mini experiment.
  Standard rates are $2.50 / million input, $0.25 / million cached input, and
  $15 / million output tokens. The public configuration remains GPT-4.1 mini;
  enabling the larger model needs explicit owner selection and costs more per token.
  See [official GPT-5.4 pricing](https://developers.openai.com/api/docs/models/gpt-5.4).
- Current Luna standard rates are $0.20 / million input tokens and $1.20 / million
  output tokens, with cached reads $0.02 / million and cache writes charged 1.25×
  the input rate. Image inputs also consume tokens. Consult the
  [official model pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
  for current terms. Each model uses a separate cache identity. There is no automatic
  model or provider fallback, no upgrade to a more expensive model and
  no claim of demonstrated parity with Qwen3-VL-30B. The alias is provider-managed,
  not an immutable weights pin; returned model IDs are retained in the raw trace.
- Without an explicit setting, `VLM_BACKEND=api` preserves the original HF route.
  The allowed `VLM_MODEL` is
  `Qwen/Qwen3-VL-8B-Instruct:featherless-ai`; arbitrary model/provider endpoints
  are not a supported fallback. This backend uses `HF_INFERENCE_TOKEN`, not
  `OPENAI_API_KEY`. Merely adding either secret does not select a provider.
- For that HF backend, set `HF_INFERENCE_TOKEN` as a Hugging Face **Space secret**, never in source,
  a public variable, browser code, logs or an uploaded `.env`. Use a fine-grained
  token with the inference-provider permission needed for requests, not the broad
  owner/admin token used to deploy the repository.
- Neither backend sends credentials to the browser, follows redirects, or silently
  switches providers. Each transport attempt reserves shared allowance first;
  bounded retries and JSON repairs count too. Response metadata includes provider,
  model, token usage where returned, and safe failure codes, not secret values.
- Initial application limits are `DEMO_MAX_RUNS=5`, `DEMO_MAX_REQUESTS=120`, and
  `DEMO_MAX_REQUESTS_PER_RUN=32`. These are shared demo limits, not an allowance per
  visitor. They do not automatically refill each day. Only the owner should change
  the configured budget after reviewing usage.
- `DEMO_USAGE_DB` defaults to `/tmp/videomasys-usage.sqlite3`. Its SQLite accounting
  survives only while that file survives; Space/container replacement can reset
  the default ephemeral ledger. Durable storage must be explicitly configured if
  the budget must persist across replacement. Do not mistake an in-container
  counter for durable account-wide enforcement.
- These application limits are **not a provider-enforced billing cap**. Requests
  consume different numbers of tokens; failed/timed-out requests may already have
  reached the provider, and activity outside this app is not covered. Account-level
  payment and provider spending controls remain the owner's responsibility.

### Optional local GPU backend

`VLM_BACKEND=local` selects the existing PyTorch/Transformers runtime instead of
the API adapter. It requires separately installed compatible `spaces`, `torch`,
`torchvision`, `transformers`, `sentence-transformers` and `accelerate` packages,
plus explicitly selected GPU hardware. The CPU `requirements.txt` intentionally
does not install that stack; changing the environment variable alone is not a
complete GPU deployment. Only this explicit branch imports `spaces` and initializes
CUDA/model weights before launching Gradio. It therefore still depends on GPU
scheduling and startup availability. `VLM_REVISION` applies only to that local
runtime, whose default Qwen weight revision is pinned.

The deployment script still permits an explicit `--hardware zero-a10g` or paid
`--hardware l4x1 --sleep-seconds 900`; L4 always requires a positive sleep timeout.
Hardware and timeout are submitted together. It never retries with a more
expensive device. Dedicated GPU billing includes startup and running idle time;
sleep is not a spending cap, and public visitors can trigger a wake. See
[GPU billing and sleep controls](https://huggingface.co/docs/hub/spaces-gpus).

Deployment uses an explicit source-file allowlist, not the entire working directory.
Local datasets, credentials, caches, and existing evaluation results are excluded.
An optional `space_library/library.json` can list owner-approved videos using entries
with `title`, relative `file`, `question`, and optional `options`. The default upload
does **not** redistribute NExTQA or EgoSchema media. The explicit `--include-library`
deployment flag includes only the media named in that manifest. This demo's
owner has selected four questions across three NExT-QA/EgoSchema videos. Examples
load the video and question in one click; inference starts only with Run analysis.
Reference answers are excluded. Media files live in the Space repository and survive
app restarts, independently of the ephemeral inference cache. Public Space visibility
also exposes repository source, bundled media, and commit history. Owners must
review redistribution permissions for the assets they choose to publish.

The deployment script creates new Spaces privately. Updating an existing public
Space requires `--allow-public`; that flag does not change repository visibility.
Visibility changes are a separate, explicit owner action.

Set `VIDEOMASYS_UI_ONLY=1` only for a local UI check. It disables inference and the
Run button, and labels the page as a preview. It never simulates a model response.

See [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).
