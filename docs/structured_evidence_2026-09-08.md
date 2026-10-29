# Consistent visual evidence in the existing QA pipeline

Local opt-in profile: `structured`, prompt version `structured-evidence-v3`,
record schema `visual-evidence-v3`.
The default remains `clean`; the public Space is unchanged. This extends the
[small hybrid](svp_hybrid_2026-09-08.md) with compact typed evidence, not new agents,
a new database, an action-memory service, or a new planning loop.

## Same workflow, explicit links between its outputs

```text
Planner → Tool agent → Cached retrieval → Visual inspection → Answer → Review
                         unchanged          typed facts       claims    checks
```

The planner and tool prompts retain the hybrid examples and example selection,
with shorter role instructions. Ordered
`locate`, `relate`, and `overview` programs, target-aware temporal ranking,
need-first selection, and the existing cache remain unchanged. Cached captions
find candidate windows; they are not visual evidence for the answer.

The new data follows a small provenance graph:

```text
Video ID
└─ Selected scene/window
   └─ Final sampled frame ID: scene + approximate seek time + image hash
      ├─ Observed fact: subject, predicate, object, need IDs, uncertainty
      └─ Answer claim: observed / inferred / limitation
         └─ Optional earlier-claim dependencies and observation references
            → Answer roots and option assessments → Per-target review checks
```

These are records and references in the current run, not a learned scene graph
or a separate graph database. The controller assigns frame and observation IDs.
Frame identity binds the video ID, scene index, requested seek position, and JPEG
bytes. The prompt receives only frame IDs, scene IDs, and approximate times in
the actual image order; file paths and hashes remain provenance metadata.

## What each existing stage returns

| Stage | Bounded output | Controller checks |
| --- | --- | --- |
| Visual inspection | Up to three atomic facts per window; unresolved needs; optional refinement frame ID | Actual supplied frame/need IDs; every requested need observed or explicitly unresolved |
| Answer | One to five claims; one to three answer roots; optional choice assessments | Valid resolved references, earlier-only inference dependencies, no orphan positive claims, planned needs addressed by answer lineage or displayed limitations |
| Review | One check per claim and supplied option | Exact target coverage without omissions or duplicates; verdict derived from all checks |

Facts distinguish states, actions, and changes. A claimed change must cite at
least two chronological samples with distinct image hashes. This is a structural safeguard,
not proof that the model recognized the change correctly. Subject and object
names are local descriptive labels, not tracked identities.

Claims separate three roles:

- `observed`: a direct description of the supplied images.
- `inferred`: a qualified conclusion with an explicit rationale; it may depend
  on earlier claims, retaining their frame and observation provenance.
- `limitation`: a specific distinction that the sampled evidence does not resolve.

The answerer may correct an inspector by citing images directly. When it reuses
an observation, the controller resolves its frame references; the model does not
copy them into a second field. An inference may inherit all its pixels through
earlier claims. Every claim must resolve to nonempty, active visual provenance;
unknown or superseded references remain invalid. Raw claim fields are preserved
alongside the separate controller-derived provenance map used by the reviewer.

Direct observations and limitations have schema-enforced empty dependencies and
empty rationale fields. Only inference claims carry a reasoning bridge. This is
expressed through disjoint typed variants in the strict output schema, not only
natural-language instructions. The OpenAI Docs skill informed this use of
[schema-native constraints](https://developers.openai.com/api/docs/guides/structured-outputs);
it does not make those claims visually true.

All limitation claims are displayed and reviewed, so they are explicit roots by
type rather than discarded as unused. Every planned need must reach an answer
root or a displayed limitation; mentioning it only for an alternative choice
does not satisfy coverage. Positive claims still must reach the answer or an
option assessment.

The controller renders the answer from the selected claim statements and derives
scene citations from direct and inherited frame references. There is no separate
free-text answer generation that can silently introduce different facts.

## Choices and review stay consistent

The existing preliminary choice call remains, but its result is a suggestion,
not a locked answer. The structured answer may choose a better-supported supplied
choice or return an explicit limitation when none is established.

Every option is assessed once in its supplied order using just two labels:

- `supported`: observed or inferred claims provide evidence for this answer.
- `unsupported`: the available evidence does not support it; this does not mean
  the option is impossible or the action never happened. Claim references may be empty.

Selection is separate: `selected_choice` identifies the supported option that
best answers the question. Other options may also have support. The selected
option's supporting claims must belong to the answer lineage. A limitation
cannot supply positive support, but an explicit limitation allows no selection
when evidence is insufficient or does not distinguish the alternatives.

Review checks every answer claim and option, including dependencies and cited
pixels. It does not independently certify every unused inspector note. The
`option_N` target is explicitly the zero-based Nth comparison, whose order is
validated. Each check is simply `pass` or `revise`. Review evaluates whether the
claim or option assessment is justified as written: an honestly `unsupported`
option can pass. A selection error belongs to the selected option's check.
The controller accepts only when every required check passes. A missing target
triggers validation repair, not silent acceptance.

One answer correction and final review are available using the same evidence.
Each `revise` reason must say what to change and why, rather than merely saying
"no evidence." A final `revise` check leaves the result unverified. A properly
worded insufficient-evidence answer can pass review but is displayed as
**Insufficient evidence**, not as a verified substantive answer.

### V3 simplification across roles

The working rule is: keep structure that links the question, retrieved evidence,
and conclusion; remove extra classification and repeated instructions.

- Planner: short visual questions and only necessary dependencies.
- Tool agent: the same scoped `locate`/`relate`/`overview` logic, paired SVP
  demonstrations, and final `inspect`; no extra steps or agents.
- Inspector: visible facts and specifically missing information.
- Answerer: a few linked observations, inferences or limitations; binary choice
  support and a separate selection.
- Reviewer: pass or a concrete correction, with the existing one-revision cap.

Caption and preliminary-choice prompts were already short and remain unchanged.
Observed/inferred/limitation and state/action/change distinctions remain because
they describe how evidence is used, not degrees of confidence. Provenance,
coverage, temporal ordering, source scoping and spending limits are not relaxed.
The binary labels are reflected in both prompts and the generated output schemas,
following [official OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs)
on keeping schemas aligned with application types.

V3 has offline regression checks for a supported selected answer with unsupported
alternatives passing review, and for a real review failure remaining unverified
after the single revision. No new live model requests or deployments were made
for this simplification. The live results below are preserved v1/v2 history, not
measurements of v3; both old evaluation runners remain version-pinned and closed.

## Refinement uses the actual final image pack

The visual budget remains at most three selected windows and 60 seconds of
selected spans, initially sampled at six frames per window. There is at most one
additional focused inspection for the analysis.

When refinement is requested, the controller samples four close-up times around
the requested frame, retaining the original first and last context samples.
The inspector sees this final, at-most-six-frame pack. Its new records replace
the earlier window observations; the same final files and IDs then go to the
choice, answer, and review stages. Superseded observations remain in the trace,
but cannot be referenced as active answer evidence. Retained context IDs remain
valid; removed interior frame IDs do not. Identical JPEG bytes within a window
are deduplicated before inspection, even if their requested seek times differ.

This fixes the earlier mismatch where a refinement inspection could describe a
12-frame grid while the downstream answer saw a different six-frame subset.
It does not increase the final frame or selected-span budget.

The structured profile also uses a fixed 512×384 thumbnail bound across image
calls. This prevents the provider adapter from encoding an inspection's six
images at one size and the answer's eighteen images at another. The default
clean/SVP adapters retain their prior batch-dependent sizing. Stored frame hashes
identify source JPEGs; adapter wire hashes identify the actual encoded API images.

## Scope and limits

This improves traceability and cross-stage consistency, not semantic certainty.
A model can still misidentify an object, invent an action, assign an inappropriate
need ID, or accept an unsupported claim. Hashes prove which stored image bytes
an ID refers to; they do not prove an interpretation of those pixels.

Scene captions and segmentation are unchanged: captions remain fallible and
windows are uniform temporal slices. Retrieval can still miss decisive evidence.
Sparse samples do not establish absence, complete activity coverage, precise
event boundaries, or continuous object identity. Seek times are approximate,
not measured decoded-frame timestamps. This profile adds neither tracking,
segmentation masks, audio understanding, nor persistent action/object memory.

The reusable boundary is evidence plus provenance, not the multiple-choice
answer schema. A future segmentation result could attach a mask to a frame ID;
a tracking result could link detections across those frames; an editing plan
could cite source intervals. Each would need its own typed result and validation,
while reusing the same evidence identifiers and observed/inferred distinction.
Those task-specific results and tools are not implemented in this QA change.

No provider, model tier, public configuration, or usage allowance is changed by
selecting this local profile. Existing JSON-repair and request limits still apply.

## Validation and evaluation status

### Initial v1 live tests: two retained failures

Both known regression cases used GPT-4.1 mini and the same immutable cached
inputs as the historical hybrid-v3 controls. The hybrid planner/tool prompts and
retriever retained their exact historical hashes. Selected scene indices matched
the historical controls in these executions. This is not an unseen benchmark or
a fresh paired comparison.

| Case | Selected scenes (zero-based) | Historical hybrid | Structured v1 | API attempts |
| --- | --- | --- | --- | --- |
| Gift, multiple choice | 0, 1, 3 | Completed in 39.2s | Stopped at Answer in 45.9s | 9 |
| Kitchen, open-ended | 2, 4, 8 | Completed in 29.9s | Stopped at Answer in 38.9s | 7 |

In both v1 runs, the first answer tried to attach dependencies to direct observed
claims. The one format repair fixed that issue but exposed another: an orphan
limitation in the gift answer, and duplicated observation/frame references that
did not agree in the kitchen answer. **Neither run produced a final answer or
reached review.** Correct-looking raw drafts are not successful results.

The batch is closed at two sessions and 16 attempts. Provider usage was 69,090
input tokens (16,384 cached) and 4,778 output tokens, with zero reasoning tokens.
These are token counts, not an account billing total. The prior ledgers remain
closed at 2/17 and 6/54 sessions/attempts, and all control file hashes remained
unchanged during both runs. See the
[closed protocol](structured_evidence_evaluation_protocol_2026-09-08.md).

Actual traces and final-frame contact sheets are retained in
`artifacts/structured_evidence_2026-09-08/structured__structured-evidence-v1__sofa_mc`
and `structured__structured-evidence-v1__kitchen_open`. Source snapshots record
the tested v1; current v2 is not retroactively attributed to those runs.

### Visual audit and v2 correction

The main interpretations remain defensible from the images: unwrapping the gift
and washing kitchenware. However, the typed inspector still made errors:

- Gift: it placed the boy in the empty-room frame at ~0.25s, upgraded already
  holding the gift into observed pickup, and the draft treated all four other
  purposes as contradicted without incompatible evidence.
- Kitchen: it described tan-bowl handling near the rack at ~84.15s as red-plate
  scrubbing, cited a rack/hand view at ~175.85s for rinsing, and added an
  unestablished cleaned condition. Object continuity into the later vessel was
  also not demonstrated.

The corrected v2 normalizes references instead of requiring duplicated frame
lists, puts direct-versus-inferred dependency rules in the wire schema, treats
displayed limitations as explicit graph roots, and removes duplicate image
bytes. A wire-hash audit additionally found that v1's sofa inspections and answer
used different resizing despite referencing the same files (kitchen matched
because its images were already smaller). V2 fixes that opt-in encoding bound;
the same-files v1 test alone did not establish identical API image bytes.
V2 adds a small fictional complete-format example and clarifies that
non-winning choices are not necessarily contradicted. These are general
contract corrections, not benchmark-specific answer patches.

The v1 failures were followed by the owner-requested v2 contract recheck below.
The profile remains local; no public deployment or visual-quality tuning was
authorized by this recheck.

Offline checks cover stale frame and observation rejection, answer-lineage need
coverage, choice ordering and support, exact review coverage, same-frame
refinement, the single-correction bound, and profile isolation. V1 passed 1,098
offline tests before live evaluation, which illustrates why synthetic contracts
alone are insufficient. V2 adds normalized-provenance, wire-schema variant,
displayed-limitation, and duplicate-pixel regression tests. The final v2 suite
initially passed **1,145 tests in 30.61 seconds**; Ruff passed for all changed Python files.
The adapter tests compare actual encoded JPEG bytes across six- and eighteen-image
packs, and an offline encoding check on the real retained gift frames matched all
18 hashes. These checks made no HTTP requests and do not validate a VLM answer.

### V2 live contract recheck

On the subsequent request to "check it", the unchanged v2 pipeline was run once
on each known case, using the same cached inputs and pinned GPT-4.1 mini. The new
batch was capped at two sessions / 24 actual HTTP attempts / 12 per case. It is
now **closed at two sessions and 20 attempts**. No earlier ledger was reopened.
Perception accuracy was explicitly out of scope; application checks and model
reviews were not bypassed or forced to accept.

| Case | Execution | QA time, cache already built | HTTP attempts | Final review |
| --- | --- | --- | --- | --- |
| Gift, multiple choice | Completed | 58.209s | 12 | Unverified |
| Kitchen, open-ended | Completed | 29.055s | 8 | Accepted |

The gift answer selected "Unwrap it". Its initial claim IDs used the reserved
`obs_` prefix; the existing one-format-repair path corrected this. The reviewer
then rejected four alternative-option assessments, and the bounded answer
revision did not resolve them. The controller correctly left the final result
unverified. This is successful execution, not a claim of clean first-pass output
or accepted reasoning. The kitchen answer described cleaning kitchenware and
completed without schema repairs; its review accepted all three targets.

An independent read-only audit verified:

- All inspection records validate against their actual requested frames and needs.
- Accepted claim graphs validate; resolved citations and rendered answers match
  their source records. Superseded observations are not active evidence.
- Both gift reviews cover all nine targets; the kitchen review covers all three.
  Controller verdicts and the one-answer-correction limit match the records.
- All 20 recorded request payloads, schemas, and encoded-image hashes reproduce.
  Final image packs are identical across the relevant inspection/answer/review
  calls: 17 distinct gift frames and 18 kitchen frames, with one refinement each.
- Source snapshots, pinned model, request accounting, and historical control
  hashes match. All 20 attempts returned HTTP 200, with zero reasoning tokens.

The earlier fatal dependency/duplicated-reference/orphan-limitation failures did
not recur. One avoidable claim-ID naming repair remains; no runtime or prompt
changes were made during this check. Two successful cases demonstrate these
paths working, not statistical reliability across arbitrary questions.
Both runs exercised dependency-only inference, but neither exercised an
observation-only claim or a displayed limitation root. Those two branches remain
offline-tested rather than live-demonstrated by this pair.

Runner: `scripts/check_structured_v2.py`. Raw bundles, reports, manifest, source
snapshots, and closed ledger: `artifacts/structured_recheck_2026-09-08/`.

### Test pruning and current scope

The 1,145 figure was the full repository suite, not this feature's tests or model
calls. The structured change added 147 collected cases: 145 in four new files and
two profile variants in an existing runtime test. On review, 45 cases were removed:

- 26 repeated schema checks, abstract-base checks covered by concrete variants,
  and basic library-validation cases. The three actual model-response schemas
  still cover their nested records.
- 17 redundant image-adapter/default-profile cases and repeated invalid-input
  variants. Exact encoded-image equality and distinct invalid-bound branches remain.
- Two repeated evaluator cases. Runner-specific credential, spending, immutable
  artifact, source-freezing, and closed-batch guards remain.

At the time of pruning, the four focused files contained **100 passing cases (3.24s)**. Pruning changed
only tests and documentation, not runtime behavior or validators. Original test
files can be recovered from `artifacts/test_pruning_2026-09-08/before.json`.
These are offline checks; they do not run with a user's analysis or add inference
latency. No additional paid model calls were made for this cleanup.

Per the user's direction, perception/hallucination errors are now treated as
out-of-scope model-quality limitations for this iteration, not architecture
acceptance blockers. The retained failure records are not rewritten. Remaining
validation work concerns execution and cross-agent contracts, not eliminating
all visual mistakes through more prompts, agents, or validators.

Implementation: [visual evidence models](../src/videomasys/domain/visual_evidence.py),
[structured pipeline](../src/videomasys/zerogpu/structured_pipeline.py), and
[shared grounding prompts](../src/videomasys/zerogpu/structured_prompts.py).

### Subsequent v3 benchmark

The later [20 NExT-QA + 20 EgoSchema live evaluation](benchmark_40_2026-09-08.md)
tested the simplified `structured-evidence-v3` profile without changing the
public Space. Only 18/40 analyses completed; 6/20 NExT-QA and 7/20 EgoSchema
answers matched their references. Sixteen of the 22 execution failures involved
claim connections or planned-need coverage. This broader check exposes reliability
problems not established by the historical two-case v2 recheck above.
