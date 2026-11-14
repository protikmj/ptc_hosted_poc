# Hosted PTC travel-budget demo

A small Python PoC for a 20-minute presentation, with direct tool calling and
OpenAI-hosted Programmatic Tool Calling over identical synthetic data.
Nothing calls the API on import. Generated JavaScript is **never executed locally**.

## Verification status — 14 November 2026

- Official [PTC guide](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling)
  documents the hosted V8 runtime, application-owned function execution, context
  isolation of intermediate values, and Responses continuation protocol.
- Installed and inspected Python SDK **3.50.0**: `ProgrammaticToolCalling`,
  `allowed_callers`, `output_schema`, program items and caller routing exist.
  [Official SDK source](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/tool_param.py).
- The guide's example uses `gpt-6-astra`. Its
  [model page](https://developers.openai.com/api/docs/models/gpt-6-astra) lists that
  identifier as its only snapshot/alias, but omits PTC from its tool list.
  No dated immutable snapshot is documented there. Both paths use that same ID,
  and audits record returned IDs; this cannot prove an alias never changed underneath.
- **Live hosted-PTC smoke test passed with `gpt-6.1-sol`.** It returned the exact
  answer 70 using three Responses requests and two Python tool invocations,
  with 1,417 API-reported total tokens. Generated programs, caller routing and
  completed hosted output were observed. Databricks execution remains untested.
- The user-selected default is now `gpt-6.1-sol`; `OPENAI_MODEL` can override it.
  Its [model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
  lists the same identifier as its snapshot/alias. The live smoke verifies PTC
  access for the supplied account even though the model page's tool list omits it.
- Local validation: ten tests cover deterministic data, exact arithmetic,
  validation, usage accounting, failed-run reporting, SDK serialization with a
  mock transport, alternating order, JSON/CSV export and PNG generation.
  The Plotly check covers dataset switching, failure markings, missing usage and HTML serialization.
  Dependency checks, compilation and the notebook with live actions disabled also passed.

The smoke test checks generated `program` items, `caller`-linked Python tool
invocations, completed `program_output`, exact result 70 and API usage. A passing
test is runtime evidence for the protocol; keeping intermediate data outside
model context is the documented service contract, not something visible from
client-side token counts alone. It does not claim the data stays outside OpenAI.
If hosted PTC is rejected, stop: the smallest alternative is a purpose-built
Python aggregation tool exposed through ordinary function calling. That would
demonstrate compact outputs, **not** dynamic hosted PTC. No fallback is implemented.

## Live results — `gpt-6.1-sol`

Three repetitions per approach, alternating order, using the same model and low
reasoning effort. These are measurements from this session, not example numbers.

| Dataset | Approach | Correct runs | Median total tokens | Median seconds |
|---|---|---:|---:|---:|
| 20 employees × 100 expenses | Standard | 0/3 | 67,390 | 41.72 |
| 20 employees × 100 expenses | Hosted PTC | 3/3 | 2,925 | 14.75 |
| 4 employees × 5 expenses | Standard | 3/3 | 2,427 | 6.80 |
| 4 employees × 5 expenses | Hosted PTC | 3/3 | 2,399 | 9.04 |

Full-data standard runs returned inaccurate monetary amounts; they did not fail
with API errors. No percentage-savings claim is made for that experiment.
Small-data median token reduction was **1.15%**, while PTC took longer.
Standard runs 2 and 3 benefited from cached inputs; PTC reported none. These
token totals therefore do not establish a corresponding cost reduction.

The local notebook shows an 8 KB metrics-only summary automatically when the
ignored `artifacts/presentation/dashboard_summary.json` is present. It is kept
local; raw requests, responses, prompts and tool results also stay out of Git.
A fresh clone needs its own live benchmark or copied saved results before the
chart can display. The notebook prints this clearly instead of silently showing
nothing.

Full audit files (local only, excluded from the GitHub repository):

- [Smoke](artifacts/live-20261114T053905774830Z/smoke.json)
- Full data: [JSON](artifacts/live-20261114T053905774830Z/full/results.json),
  [CSV](artifacts/live-20261114T053905774830Z/full/runs.csv),
  [chart](artifacts/live-20261114T053905774830Z/full/tokens.png)
- Small data: [JSON](artifacts/live-20261114T053905774830Z/small/results.json),
  [CSV](artifacts/live-20261114T053905774830Z/small/runs.csv),
  [chart](artifacts/live-20261114T053905774830Z/small/tokens.png)

Set notebook `SAVED_RESULTS` and optionally `SAVED_SMALL_RESULTS` to full or
small `results.json` paths to show those complete, explicitly labelled runs.
Copy the chosen files to Databricks, or enable a live benchmark to populate the chart.

## Local setup and commands

Python 3.10+ (local validation used Python 3.14). From the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Supply `OPENAI_API_KEY` through your existing environment/secret manager. To enter
it without shell-history storage or terminal echo, in bash/zsh:

```sh
read -s OPENAI_API_KEY
export OPENAI_API_KEY
```

Paste the key at the hidden prompt and press Enter. Do not put it in source files,
notebook output, command arguments or committed configuration.

Run actions separately; each live command consumes API credits:

```sh
# First: two trivial tools, maximum six requests, no retries.
python -m ptc_demo.runners smoke

# Inspect artifacts/smoke.json and the generated program before continuing.
python -m ptc_demo.benchmark single --approach standard
python -m ptc_demo.benchmark single --approach ptc

# Three repetitions per approach, alternating A/B then B/A then A/B.
python -m ptc_demo.benchmark benchmark --out artifacts/full
python -m ptc_demo.benchmark benchmark --small --out artifacts/small

# Explicit historical fallback (no API key or live calls needed).
python -m ptc_demo.benchmark show --out artifacts/full/results.json
```

Use a new output directory for each repeated benchmark to avoid overwriting its
audit. Single-run and smoke paths can be changed with `--out`. Set `--model` on
**all** commands if choosing another officially supported model. The benchmark
requires a successful smoke artifact for that model (`--smoke-artifact PATH`).
Check it again close to the presentation. Never interpret missing credentials as
evidence that PTC itself is unavailable.

## Databricks

1. Upload/clone the entire project into a Git folder or workspace folder.
2. Import `notebooks/ptc_demo.py` as a Python source notebook; cell markers are included.
3. Install `%pip install -r /Workspace/.../ptc_test/requirements.txt` in a setup cell;
   restart Python if the runtime requests it.
4. Set `PROJECT_ROOT` to the project folder. Set the secret scope/key in the
   configuration cell, or supply an existing `OPENAI_API_KEY` environment variable.
   The code passes `dbutils.secrets.get(...)` directly to the client without printing it.
5. Set `ARTIFACTS` to a writable workspace or Volume path. Enable `RUN_SMOKE`, run
   configuration and smoke cells, and inspect the result before enabling `RUN_SINGLE`.
6. Enable `RUN_BENCHMARK` or `RUN_SMALL` only for the explicit repeated experiments.
   Leave them false during rehearsal unless you intend to pay for another run.

Default flags are all false, so Run All does not spend credits. The notebook's
setup and display helpers are Databricks-specific; the implementation modules
and CLI run locally. Databricks secret permissions, outbound API access, dependency
compatibility and rendering still need validation on your chosen cluster.

### MLflow

The notebook uses the Databricks-provided MLflow tracking server and active
notebook experiment. Each smoke, single, benchmark, or explicitly loaded saved
result is logged as a run. Compare `approach`, `dataset`, `model`, and
`repetition` alongside correctness, API request count, Python tool-call count,
elapsed time, and input/output/total token counts in the experiment's **Runs** view.
Cached-input token counts are omitted from new MLflow runs and traces.
Incomplete usage is marked and token metrics are omitted rather than recorded as
zero. Raw result JSON and tool payloads are not uploaded as run artifacts.

When live actions are enabled and the cluster has MLflow 2.22 or later, the
notebook wraps each run in a manual parent trace. In the experiment's **Traces**
view, the waterfall shows the prompt, nested OpenAI requests, parallel Python
tool spans with their arguments and outputs, and the final model response.
Hosted JavaScript execution is part of the OpenAI response span because the API
does not expose a separate runtime span. Traces can contain prompts, model
responses and tool data; this demo's dataset is synthetic. Raw result JSON is
not uploaded as a run artifact. Older MLflow runs and traces remain unchanged.

## Interactive presentation view

The notebook now displays a dark Plotly dashboard: exact-answer counts, median
tokens and latency, plus every run in execution order. Hover for input/cached
and output/reasoning details, request counts and tool calls. Incorrect or
incomplete runs are hatched; the existing correctness gate controls savings claims.
Switch workloads with the dropdown; axes rescale for the selected dataset.

For the already measured results, set these in the notebook configuration cell
(after copying the artifact folder to Databricks):

```python
SAVED_RESULTS = str(ARTIFACTS / 'live-20261114T053905774830Z/full/results.json')
SAVED_SMALL_RESULTS = str(ARTIFACTS / 'live-20261114T053905774830Z/small/results.json')
```

Run setup/configuration and the saved-results cell. No key or paid run is needed.
The same dashboard is generated after live benchmarks. `dashboard.html` is
exported beside the selected results with Plotly JavaScript embedded for offline
use. The notebook uses `displayHTML`; re-install `requirements.txt` to add Plotly.
The existing static PNG remains available for slides.

[Open the exported full/small dashboard](artifacts/live-20261114T053905774830Z/dashboard.html).
HTML generation and figure data are tested locally; interactive rendering inside
Databricks still needs a cluster check. Implementation follows the official
[Plotly HTML export](https://plotly.com/python/interactive-html-export/) and
[Databricks displayHTML example](https://kb.databricks.com/en_US/visualizations/save-plotly-to-dbfs).

## What is measured

The fixed-seed dataset has 20 employees × 100 compact expense records; the small
version has four × five. Both have over/under-budget cases. All calculations use
integer USD cents. `data.py` independently computes ground truth; validation
rejects missing/duplicate employees, wrong names, floats and inaccurate amounts.

Both routes receive the same question, rules, function/output schemas, model,
low reasoning effort, parallel-calling setting and structured final-answer schema.
Only the orchestration instruction and PTC routing differ. Direct calls receive
full results and are not restricted to one call per turn. Independent Python
calls use a small thread pool. The PTC prompt contains no aggregation program.

Every response (including incomplete responses), generated program, tool argument,
tool reply, usage object, returned model ID, setting and final answer is saved.
All response items are replayed with `store=False`, including program fingerprints
and encrypted reasoning. Nested function outputs preserve `caller` and `call_id`.
The loop continues through program-only responses until a final assistant message.

Defaults: 12 requests/run, 8,192 output tokens/request, 100 underlying tool
invocations/run, 90-second request timeout, **zero SDK or application retries**.
The smoke has lower limits. All issued requests count, even failures. API failures
without usage are marked `usage_complete=false`: reported totals are then only
known usage, not an invented estimate of billable work. An HTTP 400/401/403/404
stops the repeated benchmark. Other failed runs remain visible.

`model_requests` counts Responses HTTP requests, including hosted continuation
requests; it does not purport to count internal model inference passes. Cached
input and reasoning tokens are reported separately as subsets of input/output,
never added again to total tokens. Missing detail fields are `null`, not zero.
End-to-end time includes API waits, local tool execution and validation of the
protocol loop; final ground-truth checking and writing artifacts are outside it.

`results.json` contains raw responses, data, configuration and smoke preflight;
`runs.csv` contains per-run metrics; `tokens.png` charts actual reported totals.
The smoke and demo single runs are separate from benchmark medians and their
usage is saved separately (the smoke is also embedded for provenance).
Medians include failed attempts. Savings require correct answers, complete
reported usage, equal run counts and matching returned model IDs. Missing usage
suppresses affected medians. Failed/unknown chart bars are labelled.

The full and small workloads are separate experiments. Alternating order reduces
order effects but does not eliminate caching or service variability. Three runs
are a demonstration, not a statistically robust performance study.

## 20-minute presentation

| Minutes | Show |
|---|---|
| 0–3 | Direct calls versus a dynamically generated orchestration program |
| 3–5 | Dataset, cents, budget rules; successful smoke preflight |
| 5–8 | Standard live run and complete tool outputs |
| 8–12 | Hosted run, generated JavaScript, runtime-routed calls and emitted result |
| 12–14 | Independent correctness check before interpreting usage |
| 14–17 | Three-run comparison, medians and token chart (use saved real results if slow) |
| 17–20 | Small-data experiment, caveats and questions |

Rehearse and save genuine results before the talk. Set `SAVED_RESULTS` to that
file for an explicitly labelled historical fallback.
Large direct-tool outputs may make live execution slower than the allotted slot.

This workload favors aggregation. SQL or a purpose-built backend aggregation
tool could also return compact data. PTC's distinctive benefit is dynamically
composed orchestration. Token reduction is not automatically an equal reduction
in cost or latency; this demo makes no cost claim.

## Files and checks

- `data.py`: deterministic records and independent answer validation.
- `tools.py`: three plain Python tool implementations and schemas.
- `runners.py`: shared API loop, secure client creation and two-tool smoke test.
- `benchmark.py`: explicit runs, alternating benchmark, audit files, table/chart.
- `visuals.py`: Plotly comparison view shared by live and saved notebook results.
- `artifacts/presentation/dashboard_summary.json`: local metrics-only fallback; excluded from GitHub.
- `tests/test_core.py`: offline data, arithmetic, validation, usage and protocol checks.

Tests use simulated protocol responses only where a live API would otherwise be
required. They are not evidence of live hosted execution. Artifacts are ignored
by Git because raw API responses include opaque encrypted content. Cloning gets
the runnable code and notebook; generate your own results or explicitly transfer
the local saved artifacts separately for presentation backup.
