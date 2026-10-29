# Single-video editing extension — 2026-09-08

This adds an independent Edit view to the Gradio demo. Ask retains the published
GPT-5.4 / `structured-evidence-v7` QA pipeline, library and usage settings.

## Data and execution

One typed `EditDocument` is shared by AI proposals, browser changes and rendering:

```json
{
  "version": "1.0",
  "source_id": "<server-computed SHA-256 of the video>",
  "clips": [
    {"id": "clip_1", "start": 12.0, "end": 18.0, "label": "Selected moment", "mute": false}
  ],
  "title": ""
}
```

Clip order is output order; start/end are source seconds, with an exclusive end.
There are no source paths, shell commands or executable model code in this document.
Manual operations change only this document. Browser sequence playback seeks over
excluded ranges; it is a quick approximate preview, not a pre-rendered seamless file.
FFmpeg creates the deterministic preview/export when requested.

The trim view uses full-source coordinates and a fixed, viewport-fitted strip.
The selected clip's kept range is highlighted; left/right exclusions remain visible
in gray. Trimming changes the range, not the strip's width or time scale. A compact
sequence row separately shows output clip order. The source strip can be scrubbed
while paused to inspect excluded footage without adding it to the sequence.

The new AI path is:

```text
Instruction → evidence plan → typed cache retrieval → sampled-frame inspection
                                                      ↓
                          frame-linked cut proposal → validated edit document
                                                      ↓
                                           manual timeline → preview/export
```

It reuses the QA caption/vector cache and existing typed retrieval/inspection
machinery, including its bounded repairs/refinement. It does not run the QA
answer selector or verifier. Clips must resolve to active sampled-frame references
inside inspected windows; broad observations are optional and must fit the proposed
clip if cited. A no-support or invalid proposal fails explicitly and does not replace
the user's current timeline. Actual prompts, tools, evidence, repairs and usage are
available in the editing trace. No canned or benchmark-reference cuts are returned.

The new editing prompt has its own version (`editing-evidence-v2`). Following the
OpenAI Docs skill's [GPT-5.4 prompt guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4),
the prompt stays compact with explicit output/evidence boundaries and adds repair
detail for a measured failure; it does not increase reasoning effort or change QA.

## Interaction and bounds

- Single source/track, draggable trim handles, numeric in/out, split, reorder,
  delete, undo/redo, reset, clip labels, mute, and a title on the first three output seconds.
- Source: existing 100 MiB, 180-second, 4K bounds. Editable timeline: 0–12 clips,
  each at least 0.1 seconds; total at most 180 seconds. Rendering requires a nonempty cut.
- AI: up to three nonoverlapping clips within a user-selected maximum of 5–60 seconds.
  This is a maximum, not an obligation to pad the output. AI operates on sampled frames;
  it cannot guarantee frame-perfect boundaries or that no relevant event was missed.
- Preview: 640×360. Export: 1280×720. H.264/AAC, 30fps, original aspect ratio letterboxed;
  original audio retained unless a clip is muted. Titles are literal text, not filter code.
- No model calls for manual operations, playback, preview rendering or export.
  Find clips shares the existing hosted-model session/request limits with QA.
- Bounded CPU rendering: one active renderer, cancellation and a 240-second deadline,
  two encoding threads, 128 MiB output/intermediate bounds, 512 MiB/32-file cache ceiling.
  A full cache rejects new renders rather than deleting another user's files.
- Server-owned source/hash validation is repeated for renders and cache hits. Client
  URLs, claimed durations and paths are not trusted. Sources are never overwritten.
- A per-session generation/job guard prevents stale source/operation responses from
  replacing a newer editor state; changing a cut invalidates its old download/preview.

No transitions, music library, speech captions, multi-track compositing, multi-source
projects or durable project storage are included. Session/cache files are temporary;
download the MP4 before leaving.

## Verification record

The renderer has focused synthetic-video/audio checks for trimming, reordering,
duration, selective muting, literal titles, output size, cache identity, cancellation
and source protection. The component is exercised in Chromium with actual Gradio,
real library media, drag/numeric edits, undo/redo, empty/reset, source changes and playback.
The existing QA browser smoke also passes, including expand/collapse, error recovery,
mobile scrolling and reaching the final evidence outputs. These are UI fixtures,
not new QA accuracy measurements.

The first real editing smoke on the gift video failed its cut contract after one
repair (13 HTTP calls, 39.64s, $0.1012305). A cited broad observation touched the
exclusive cut endpoint; widening the cut in repair exceeded the 30-second maximum.
No cut was applied. Its full trace is preserved in
`artifacts/editing_smoke_2026-09-08/`. The correction keeps containment checks and
makes optional observation links and all-violation repair feedback explicit.

The one corrective live run completed on its first cut-proposal attempt: a single
17.5048-second clip, source 54.2644–71.7692 seconds, with audio preserved. Total run
time was 35.13 seconds (including a fresh index); 12 HTTP calls cost $0.08725.
Both runs together cost $0.1884805, with no unknown/reserved charges remaining.
This is one exposed development example, not an editing accuracy benchmark.
Records are under `artifacts/editing_smoke_2026-09-08/retry-v2/`. Its broad
source-drift check noticed concurrent UI development, so the runner exited nonzero;
the editing request itself completed. Inference code and the original run records
remained unchanged during that corrective test.

The real AI cut rendered in 3.42 seconds: H.264, 1280×720, 17.522-second container
(17.500-second video), with non-silent AAC stereo audio. Its request/frame/ledger
and export checks are in `retry-v2/live-audit.json`. Visual review confirmed the
unwrapping action; the cut includes follow-through and is still a rough edit.

Final local verification:

- 160 focused tests passed; one opt-in browser test skipped in that invocation.
  The component's separate real Chromium invocation passed both component checks.
- Full-app browser smoke passed all 15 checks, including real source loading,
  numeric trim, split/reorder/undo, title/mute/playback, a downloaded 10-second
  720p export, stale-download invalidation, Ask/Edit preservation, new-source reset,
  and mobile bounds without JavaScript errors.
- Desktop/light and mobile/dark screenshots were visually inspected. After waiting
  for resize layout, the mobile timeline and ruler remain visible. An independent
  browser recheck confirmed no persistent blank timeline after source/viewport changes.
- The existing QA UI smoke passed its streamed-stage, scrolling and error-recovery
  checks using explicitly labeled authored fixtures. No new QA model calls were made.
- Ruff passed on all changed Python modules, tests and the browser smoke script.

Browser reports/screenshots are under `artifacts/editor_ui_2026-09-08/`.

## Publication

Published to the existing public `lqh52/videomasys-demo` Space as
`57f4e3727629306505a1853d3d934c7cf0c86da9`, with compare-and-swap parent
`6fd6f3d3f0f7f523a5b7beb1023a1714594b04b0` (the prior v7 release/rollback source).
Only seven allowlisted app files changed: the Space README, four new editing modules,
and the existing UI/style modules. All 63 other allowlisted files—including QA
inference sources and library assets—were unchanged. All 70 deployed file hashes
match the frozen local inputs. No artifacts, labels, tests, `.env` or credentials
were uploaded.

Hub metadata confirms public visibility, `RUNNING`, and the same source/container
revision. Fresh runtime logs include Gradio's startup-ready message and no traceback.
CPU Basic, one replica, 172800-second sleep, OpenAI/full pinned GPT-5.4,
structured QA profile, secrets and quotas were preserved. No hardware or billing
settings changed. The local editing-evaluation dollar cap was not deployed;
existing shared request/run limits remain the public cost control, not a dollar cap.

Normal HTTPS to the canonical hosted page still fails certificate validation from
this workspace. No TLS bypass or alternate routing was used. Hosted browser and
hosted inference tests therefore cannot be claimed; verification used local real
browser/media tests, direct bounded model tests, and Hub build/runtime metadata.

[Open the Space](https://huggingface.co/spaces/lqh52/videomasys-demo) and choose **Edit**.

## Follow-up: stable source trimming

After user feedback, full-video trimming was reproduced in a real browser. An outer
edge drag hit the playhead and only scrubbed. An inner handle drag changed source
in from 0 to 6.124 seconds, but the sole clip stayed 733 pixels wide because the
sequence automatically refitted on each change. The 180-second source's right edge
was also outside the initial visible timeline.

The trim strip now uses the full source duration at a fixed viewport-fitted scale.
Gray overlays show excluded footage on either side of the selected clip, and remain
after release. The handles sit outside the kept interval in padded gutters, so they
remain reachable at both full-video endpoints and do not overlap for short clips.
Only the playhead knob captures pointer input; its line cannot intercept trim drags.
Output ordering uses a compact separate Sequence row. Mouse, touch and keyboard
adjust the same validated edit document, with one commit per completed pointer drag.
Paused inspection of gray footage changes no edits; output playback still uses kept clips.

Verification: two real Gradio/Chromium component checks passed, including mouse/touch,
both source-boundary handles, cancellation, bound clamps, unchanged rail/thumbnails,
gray-mask growth, one commit on release, undo/reorder, playback and stale generations.
The full-app smoke also passed, including a real 10-second 720p export. A further 92
renderer, UI guard, QA callback and deployment tests passed; Ruff was clean.
Desktop/light and mobile/dark screenshots were visually inspected. In a direct
full-app drag check, source width remained 706.90625 pixels while the gray regions
grew to 120 and 100 pixels. No model requests were made for this UI-only change.
Artifacts are under `artifacts/gray_trim_2026-09-08/`.

The final three-minute source check at 390-pixel viewport width kept both handles
in view and the strip fixed at 262 pixels. Dragging both endpoints updated the kept
range to 20.611–159.389 seconds; gray-region inspection showed a visible mobile
"Outside selected clip" status without changing the cut. The final component tests
passed again after this mobile-status polish.

Published revision: `01b0b295accd231099e990fda28911f802f03d4e`, parent/rollback
`57f4e3727629306505a1853d3d934c7cf0c86da9`. Only the component and Space README
changed; the other 68 allowlisted files stayed unchanged. All 70 deployed hashes
were verified. Hub reports `RUNNING` at the new revision, and startup logs show
readiness with no traceback. Public visibility, CPU Basic, one replica, sleep and
model/profile settings remain unchanged. Browser testing was local; no hosted
browser test or model request is claimed for this UI-only publication.
