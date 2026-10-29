# QA system audit — 7 September 2026

This is a diagnostic evaluation of four curated questions over three videos, not
a general accuracy benchmark. Reference answers are confined to local evaluation
code and artifacts; the deployed application receives only video, question, and
the user's answer choices.

## Method

The original runtime was archived before modification. Baseline and subsequent
runs use Qwen/Qwen3-VL-8B-Instruct in BF16 on the same local vLLM server, with real
MiniLM embeddings, actual FFmpeg frames, and no model-output fixtures. Each later
run freezes its runtime source and stores prompts, responses, validation errors,
retrieval scores, selected images, decisions, and timings. Decoding changes and
frame-sizing policies are recorded by the evaluator.

These local times exclude model startup and do not predict Hugging Face queue or
ZeroGPU latency. Within a fresh four-question run, the first Christmas question
builds the video cache and the second reuses it. Separate warm runs are labeled.

Human inspection used contact sheets from the bundled source videos and finer
sampling around relevant interactions. See the local
[visual audit](../artifacts/evaluation/video_audit.md). No synthetic imagery was
used for evidence.

## Baseline findings

| Example | Baseline outcome | Local time |
| --- | --- | ---: |
| Christmas: move to sofa | Correct choice, verifier rejected | 46.1 s |
| Christmas: return to tree | Inspector returned invalid scene IDs; no answer | 23.5 s |
| Kitchen objective | Correct, accepted | 32.7 s |
| Painting cup purpose | Correct, accepted, but claimed contacts absent from selected samples | 41.2 s |

Every baseline tool program had an unused retrieval output. For the sofa question,
`r1` was omitted from `inspect.sources`; its anchor was also incorrectly the final
window, leaving no strict subsequent hits. The planner's supposed outcome merely
repeated the movement in the question. This was not a functioning causal evidence
chain.

The sofa video visibly shows removal of wrapping and the revealed item. The boy
stands rather than sits. The return-to-tree interaction occurs near the clip end,
after the old final sample. Painting contains brief brush-to-cup contacts that
widely spaced samples often miss. Correct labels alone therefore overstated the
quality of the old system.

## Implemented changes

- Separate scope-aware planning and typed tool compilation remain model-generated.
  The planner receives the existing caption timeline as explicitly fallible hints.
- Static validation requires live dataflow to inspection and grounded dependency
  edges. Repair messages report the related violations together; no controller
  invents a replacement program or retrieval result.
- Temporal candidates include the anchor window, following SVP's coarse-window
  approach, without claiming event order inside it. Target-query cosine reranks
  eligible candidates; temporal proximity remains a separate tie-break.
- Candidate selection prioritizes evidence needs and relation outputs within three
  windows and 60 seconds. Whole-video plans require early/middle/late coverage.
- Indexing uses four endpoint-aware samples. Selected windows receive six fresh
  samples, inspected individually to reduce image/scene confusion.
- One optional close-look request samples 12 frames in a clipped ±2-second region
  inside a selected window. Final sampling retains its original first/last images
  plus four focused images, chronologically ordered. Final visual calls remain
  bounded to 18 images; no whole-video fallback is used.
- Answers select exact choice text, with an explicit deterministic index mapping.
  Scene citations are constrained to actual selected IDs. Every choice receives a
  model-generated comparison, visible in the expanded pipeline.
- Verification sees the selected images as well as the draft. It may allow a
  grounded purpose inference, but is not independent proof of correctness.
- A small repetition penalty and fresh malformed-JSON repair avoid echoing a
  truncated generation loop. Full failed output is retained in the trace.

The SVP references were its planner/reasoner separation and
`structured_vp/execution/retrieval.py`: typed location/temporal operations, anchor
inclusion, bounded selected-segment sampling, and concise final-choice prompting.
Its whole-video and fuzzy-answer fallbacks were not copied.

## Intermediate evaluations retained

| Revision | Completed | Correct | Important failure |
| --- | ---: | ---: | --- |
| Baseline | 3/4 | 3/4 | Sofa rejected; tree crashed; painting's visual claims unsupported |
| V1 | 2/4 | 2/4 | Tool repair removed required relations; image IDs used as citations |
| V2 | 3/4 | 2/4 | Wrong option index accepted; repetitive painting inspection failed |
| V3 | 4/4 | 3/4 | Painting chose a broad water-supply explanation |
| V4 | 4/4 | 3/4 | Explicit comparisons still chose the broad painting alternative |
| V5 painting-only, warm | 1/1 | 0/1 | Close-look sampling worked; function comparison remained wrong |

Three controlled prompt ablations then used the exact same V5 painting evidence:
compact visual QA, compact observations-only QA, and compact visual QA with a
general function-versus-resource distinction. All selected the correct exact
choice. The visual calls took about 0.9 seconds each locally. This is evidence of
a prompt/task-format interaction, not evidence that longer prompts are generally
worse. The final runtime separates a short visual choice-selection call from
candidate explanation/comparison. The candidate is not treated as a fact: the
answerer may mark it unsupported and the verifier may reject it.

## V6 local evaluation

V6, with a fresh cache across the four questions, completed **4/4 correctly**, with
all four accepted by the model verifier and **zero validation repairs**.

| Example | Exact choice | Retrieved window IDs | Local time |
| --- | --- | --- | ---: |
| Sofa | Unwrap it | 1, 2, 3 | 39.7 s |
| Return to tree | Continue unwrapping presents | 0, 1, 4 | 29.8 s |
| Kitchen | C is cleaning dishes | 1, 4, 7 | 46.5 s |
| Painting | To clean the paintbrush | 0, 1, 7 | 57.2 s |

The sofa evidence now covers approximately 18–72 seconds, including actual
wrapping removal. Kitchen spans the beginning, middle, and end thirds. Painting
requested the close-look region 17.75–20.0 seconds within scene 0. These are actual
retrieval/refinement results, not inserted timestamps or benchmark rules.

Some intermediate captions and observations still misname objects or overstate
actions. Final correctness on this small set does not make those descriptions
infallible. The role outputs remain visible rather than being rewritten after
comparison with labels.

Reversing all answer choices and reusing the video caches produced another **4/4
correct, accepted results with zero repairs**. Warm times were 28.5, 28.0, 25.7,
and 32.2 seconds. The same seven-module source snapshot was used in both runs.
Kitchen's correct index moved from 3 to 1, and painting's from 4 to 0; choice text
remained correct. The combined result is **8/8 on four curated questions in two
orders**, not eight independent or held-out questions.

## Hosted deployment

The first improved private Space deployment was commit
`85dc04a5f422f085a91f92a83d39a9bfef5d19ad`. Its configuration returned HTTP 200
and no longer contains the removed GPU/quota sentence. The real hosted sofa run
selected **Unwrap it**, was accepted, and needed no repairs: **53.7 seconds**
end-to-end through the Gradio API. This validates the Transformers/ZeroGPU path
in addition to local vLLM; it is a measured run, not a latency promise.

The hosted V6 painting run also chose **To clean the paintbrush**, but its verifier
**rejected the visual support** (76.7 seconds, zero validation repairs). Unlike the
local run, the inspector requested a close look around 11.95 seconds, missing the
cup interaction. Replacing the entire window's six final images with that wrong
close look also discarded useful context. This exposed a real refinement-policy
defect, not a JSON error. V7 changes the final six-image composition to retain
the original first/last samples plus four focused samples. It does not force the
answer or disable review warnings.

V7 is deployed at commit `0c213100bcdd5c63805fb720936095c32ca6f0c7`;
the Space is private, running on ZeroGPU, and its configuration returned HTTP 200.
The final patch's local painting regressions both selected the correct exact
choice, were accepted, and needed zero repairs: **51.4 seconds** in normal order
and **34.5 seconds** reversed, both warm-cache. Both used the same frozen source,
one 12-frame close look, and 18 final images. The other three questions were not
rerun on V7; their V6 runs did not invoke the changed refinement path.

The real hosted V7 painting rerun completed in **77.7 seconds**, selected
**To clean the paintbrush**, was accepted, and needed zero repairs. It again
requested the poor 11.95-second close look, but now retained the original
0.25/19.75-second context samples. Thus the sampling fix was exercised on the
actual failure path; the inspector's focus-selection weakness remains. This was
a cold caption-cache run after deployment. Hosted sofa was checked on V6 (its
non-refinement path is unchanged), and painting on V7; kitchen and return-to-tree
were checked locally, not through the final hosted deployment.

## Remaining evidence-quality limitations

A final human check of all 18 V7 local painting answer images confirms brush/cup
contact around 19.75 seconds (scene 0) and 28.63 seconds (scene 1), with palette and
painting context before/after. Scene 7 supplies painting context, not a cup-contact
observation, so citing all three scenes as direct cleaning evidence is too broad.
Cleaning is a defensible best-choice inference, but the model's added claims about
specific color changes, preventing mixing, and ensuring clean strokes are not
directly established. Some inspectors also describe one sampled interaction per
window as repeated contacts; two windows together show two contacts, not repeated
contacts within each window. The hosted explanation contains similar unverified
embellishments. The verifier accepted these, demonstrating the limitation of
using the same model for generation and review. Correct-choice counts should not
be read as fully grounded explanation scores. No explanations were overwritten
with benchmark-derived text, and these shortcomings remain visible in the trace.

## Engineering checks

After the context-preservation fix, the full suite passed **306 tests** and Ruff
reported no issues. These cover typed dataflow, temporal coverage, source/scene
IDs, exact choice mapping, failed-call traces, bounded sampling, and preservation
of original context through choice selection, answering, and verification.
Nine local browser smoke checks also passed for the existing layout, thumbnails,
run acknowledgement, error/retry flow, and narrow-screen layout. Browser fixtures
exercise the UI only and are never counted as model evaluations.

These failures are not hidden or relabeled as successes. Artifacts live under
`artifacts/evaluation/qa_system_improvement/`. In particular, successful JSON
validation and a positive model verifier are not equivalent to a correct answer.

## Reproducing local evaluation

```bash
.venv/bin/python scripts/evaluate_zerogpu_library.py \
  --engine local --phase baseline \
  --source-archive artifacts/evaluation/qa_system_improvement/baseline_source.tar.gz \
  --output-dir artifacts/evaluation/qa_system_improvement

.venv/bin/python scripts/evaluate_zerogpu_library.py \
  --engine local --phase improved-candidate \
  --output-dir artifacts/evaluation/qa_system_improvement
```

The local runner expects a Qwen endpoint at localhost:8205 and a compatible MiniLM
embedding endpoint at localhost:8206. `scripts/serve_eval_embeddings.py` provides
the latter in a separate CPU/Transformers environment. `--reverse-options` changes
option order and remaps labels only in the scorer; `--cache-root` enables explicitly
warm comparisons. Hosted evaluation uses the real private Space API and consumes
the account's ZeroGPU quota. UI fixture checks are separate and never count as
model-quality evidence.
