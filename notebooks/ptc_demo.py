# Databricks notebook source
# MAGIC %md
# MAGIC # Code Mode / Programmatic Tool Calling
# MAGIC **Question:** Which employees exceeded their Q3 travel budget, and by how much?
# MAGIC
# MAGIC Direct calls: model → Python tools → full results → model calculation.
# MAGIC Hosted PTC: model writes JavaScript → OpenAI V8 → Python tools → V8 aggregation → compact result → model.
# MAGIC
# MAGIC Python implements the tools; JavaScript orchestrates them. No generated code runs on this cluster.
# MAGIC Hosted PTC passed locally with gpt-6.1-sol. Run the smoke test below on this cluster before presenting.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup (before presenting)
# MAGIC Put this project in a Databricks Git folder or workspace folder. Install with
# MAGIC `%pip install -r /Workspace/path/to/ptc_test/requirements.txt`, then restart Python if required.
# MAGIC Set `PROJECT_ROOT` below. Use Python 3.10+; no Spark is required.

# COMMAND ----------
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path.cwd()  # Change to /Workspace/.../ptc_test if needed.
if PROJECT_ROOT.name == 'notebooks':
    PROJECT_ROOT = PROJECT_ROOT.parent
assert (PROJECT_ROOT / 'ptc_demo').is_dir(), 'Set PROJECT_ROOT to the project folder'
sys.path.insert(0, str(PROJECT_ROOT))

from ptc_demo.data import make_data, expected_answer
from ptc_demo.runners import DEFAULT_MODEL, make_client, smoke, run, save_json
from ptc_demo.benchmark import benchmark, print_table, token_chart, load_saved, require_smoke
from ptc_demo.visuals import benchmark_dashboard

# COMMAND ----------
MODEL = DEFAULT_MODEL  # Use the exact same identifier for every run.
ARTIFACTS = PROJECT_ROOT / 'artifacts'  # Or a writable /Volumes/... path.
RUN_SMOKE = False
RUN_SINGLE = False
RUN_BENCHMARK = False
RUN_SMALL = False
SAVED_RESULTS = ''  # Explicit fallback: /.../results.json from a previous real run.
SAVED_SMALL_RESULTS = ''  # Optional small results.json for the dashboard's dataset dropdown.

# Only load a key when a live action is enabled. Never display the key.
client = None
if RUN_SMOKE or RUN_SINGLE or RUN_BENCHMARK or RUN_SMALL:
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        api_key = dbutils.secrets.get(scope='YOUR_SCOPE', key='openai-api-key')
    client = make_client(api_key)
    del api_key

# COMMAND ----------
# MAGIC %md
# MAGIC ## Hosted-PTC smoke test — run this first
# MAGIC Enable `RUN_SMOKE` and rerun configuration. A pass checks generated program items,
# MAGIC routed Python calls, completed hosted output, the exact answer, and reported usage.
# MAGIC The context boundary is documented by OpenAI; token counts alone cannot prove it.
# MAGIC If the API rejects PTC, stop and inspect the audit. Do not substitute Code Interpreter.

# COMMAND ----------
smoke_result = None
if RUN_SMOKE:
    smoke_result = smoke(client, MODEL)
    save_json(smoke_result, ARTIFACTS / 'smoke.json')
    print({k: smoke_result[k] for k in ('correctness', 'usage', 'error')})
    for program in smoke_result['programs']:
        print(program)
    require_smoke(smoke_result, MODEL)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dataset: compact, synthetic, deterministic
# MAGIC All amounts are USD cents. Budgets depend on level; exceeding means strictly greater.

# COMMAND ----------
data = make_data()
print(data['config'])
print('Employees:', len(data['employees']), 'Expense records:', len(data['expenses']))
print('Employee sample:', data['employees'][:2])
print('Expense sample:', data['expenses'][:3])
print('Budgets:', data['budgets'])

# COMMAND ----------
# MAGIC %md
# MAGIC ## A. Standard tool calling
# MAGIC Independent calls may run together. Every complete tool result is returned to the model.

# COMMAND ----------
standard = None
if RUN_SINGLE:
    standard = run(client, data, 'standard', MODEL)
    save_json(standard, ARTIFACTS / 'standard-single.json')
    print(standard['answer'])
    print_table([standard])

# COMMAND ----------
# MAGIC %md
# MAGIC ## B. Hosted PTC
# MAGIC The model generates the orchestration dynamically. We supply no completed aggregation program.

# COMMAND ----------
ptc = None
if RUN_SINGLE:
    if smoke_result is None:
        import json
        smoke_result = json.loads((ARTIFACTS / 'smoke.json').read_text())
    require_smoke(smoke_result, MODEL)
    ptc = run(client, data, 'ptc', MODEL)
    save_json(ptc, ARTIFACTS / 'ptc-single.json')
    for program in ptc['programs']:
        print(program)
    print(ptc['answer'])

# COMMAND ----------
# MAGIC %md
# MAGIC ## What reaches the model?
# MAGIC Both routes send tool data to OpenAI. In PTC, `caller` routes nested results to the runtime;
# MAGIC the program chooses what to emit to the model. This is a context boundary, not a privacy boundary.
# MAGIC Below are actual returned items, not token estimates. Full traces are saved in JSON.

# COMMAND ----------
if standard and ptc:
    print('STANDARD — first full tool reply:', standard['tool_results'][:1])
    print('PTC — first runtime-routed reply:', ptc['tool_results'][:1])
    print('PTC — emitted program results:')
    for response in ptc['responses']:
        for item in response['output']:
            if item['type'] == 'program_output':
                print(item)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Correctness before savings
# MAGIC Independent Python ground truth checks identities, names and exact monetary values.

# COMMAND ----------
if standard and ptc:
    print('Ground truth:', expected_answer(data))
    print('Standard correct:', standard['correctness'], 'PTC correct:', ptc['correctness'])
    print_table([standard, ptc])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Repeated benchmark (explicit paid action)
# MAGIC Three runs per approach: A/B, B/A, A/B. Failed runs remain visible.
# MAGIC Cached tokens are part of input; reasoning tokens are part of output. Do not add them again.
# MAGIC Request count includes hosted continuation requests, which need not each perform new model inference.

# COMMAND ----------
if RUN_BENCHMARK:
    if smoke_result is None:
        import json
        smoke_result = json.loads((ARTIFACTS / 'smoke.json').read_text())
    result_dir = ARTIFACTS / ('full-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    results, result_dir = benchmark(client, data, smoke_result, MODEL, out_dir=result_dir)
    print_table(results['runs'])
    token_chart(results['runs'], result_dir / 'tokens.png')
    fig = benchmark_dashboard({'Full dataset': results}, saved=False)
    fig.write_html(result_dir / 'dashboard.html', include_plotlyjs=True)
    displayHTML(fig.to_html(full_html=False, include_plotlyjs=True))
    print('Audit:', result_dir)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Small-data comparison and limitations
# MAGIC Four employees, five records each; budgets and total-spend scale remain comparable.
# MAGIC Code generation overhead may outweigh savings for small workloads.
# MAGIC This workload favors aggregation. A SQL query or purpose-built backend tool could also return
# MAGIC compact results. PTC's distinctive benefit here is dynamically composed orchestration.
# MAGIC Token reduction does not imply the same percentage reduction in cost or latency.

# COMMAND ----------
if RUN_SMALL:
    if smoke_result is None:
        import json
        smoke_result = json.loads((ARTIFACTS / 'smoke.json').read_text())
    small_dir = ARTIFACTS / ('small-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    small_results, small_dir = benchmark(client, make_data(small=True), smoke_result, MODEL, out_dir=small_dir)
    print_table(small_results['runs'])
    token_chart(small_results['runs'], small_dir / 'tokens.png')
    datasets = {'Small dataset': small_results}
    if RUN_BENCHMARK:
        datasets = {'Full dataset': results, **datasets}
    fig = benchmark_dashboard(datasets, saved=False)
    fig.write_html(small_dir / 'dashboard.html', include_plotlyjs=True)
    displayHTML(fig.to_html(full_html=False, include_plotlyjs=True))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Explicit fallback: previously saved real results
# MAGIC Set `SAVED_RESULTS` to a real benchmark's `results.json`. These are historical measurements,
# MAGIC not a successful live demo. No example benchmark numbers ship with this project.
# MAGIC Set `SAVED_SMALL_RESULTS` too for a full/small dropdown. Hover for token breakdowns,
# MAGIC zoom to inspect runs, and use the camera button to download a PNG. HTML exports work offline.

# COMMAND ----------
if SAVED_RESULTS:
    saved = load_saved(SAVED_RESULTS)
    print_table(saved['runs'])
    datasets = {'Primary dataset': saved}
    if SAVED_SMALL_RESULTS:
        datasets = {'Full dataset': saved, 'Small dataset': load_saved(SAVED_SMALL_RESULTS)}
    fig = benchmark_dashboard(datasets, saved=True)
    fig.write_html(Path(SAVED_RESULTS).with_name('dashboard.html'), include_plotlyjs=True)
    displayHTML(fig.to_html(full_html=False, include_plotlyjs=True))
    for result in saved['runs']:
        for program in result['programs']:
            print(program)
