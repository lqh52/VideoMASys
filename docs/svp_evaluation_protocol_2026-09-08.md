# Local SVP evaluation protocol

The evaluation uses the same pinned GPT-4.1 mini snapshot, video bytes, historical
scene captions, and cached vectors for the clean and SVP candidates. It runs only
locally through the OpenAI API; the public Space and its allowances are unchanged.
Case directories and artifacts are exclusive and are never overwritten or rerun.
The tests are known regression cases, not unseen holdouts.

## Explicit amendment for the second SVP candidate

After the original four runs consumed **36 of the existing 60 HTTP attempts**,
the protocol was amended to permit one `svp-roles-v2` run for each of the same two
cases. The maximum number of sessions increases from four to six; the shared
HTTP-attempt allowance remains **60**, with no refund, reset, replacement ledger,
or automatic new batch. Each additional run is limited to **12 attempts**, or the
smaller remaining global allowance. Retries and failed requests count.

| Run group | Maximum sessions in existing ledger | Global HTTP-attempt cap | Per-run cap |
| --- | ---: | ---: | ---: |
| Original clean and SVP v1 comparisons | 4 | 60 | 32 |
| Additional SVP v2 comparisons | 6 total | 60, unchanged | 12 |

The original manifests retain their original four-session limits. V2 manifests
record the amendment, counters observed before admission, remaining allowance,
and historical v1 source hashes. V2 artifacts use the separate directory names
`svp__svp-roles-v2__sofa_mc` and `svp__svp-roles-v2__kitchen_open`; they also retain
the tested `core.py` and `svp_retrieval.py` source snapshots. All runs share
`artifacts/svp_logic_2026-09-08/usage.sqlite3`.

V2 preflight refuses a missing, reset, or exhausted ledger. It also requires the
preserved v1 case manifest. The explicit `--run` flag admits one paid run; without
it the runner validates inputs and protocol read-only, without opening owner
credentials or calling inference. These are local request limits, not an OpenAI
account billing cap or a guarantee of answer correctness.
