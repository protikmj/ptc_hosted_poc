"""Run explicitly, preserve failures, and report only measured usage."""
import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .data import make_data, expected_answer
from .runners import DEFAULT_MODEL, make_client, run, save_json

METRICS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens',
           'total_tokens', 'model_requests', 'tool_invocations', 'elapsed_seconds')


def run_row(result):
    return dict(approach=result['approach'], repetition=result.get('repetition', 1),
                correctness=result['correctness'], usage_complete=result['usage_complete'],
                error=json.dumps(result.get('error')), requested_model=result['requested_model'],
                returned_models=','.join(result['returned_models']),
                **{key: result['usage'].get(key) if key.endswith('tokens') else result[key]
                   for key in METRICS})


def summarize(runs):
    medians = {}
    for approach in ('standard', 'ptc'):
        selected = [run_row(r) for r in runs if r['approach'] == approach]
        if not selected:
            continue
        medians[approach] = dict(runs=len(selected), correct_runs=sum(r['correctness'] for r in selected))
        for key in METRICS:
            values = [None if key.endswith('tokens') and not r['usage_complete'] else r[key]
                      for r in selected]
            medians[approach][key] = (statistics.median(values)
                if all(v is not None for v in values) else None)
    returned = {tuple(r['returned_models']) for r in runs}
    comparable = (len(medians) == 2 and medians['standard']['runs'] == medians['ptc']['runs']
                  and len({r['requested_model'] for r in runs}) == 1
                  and len(returned) == 1 and len(next(iter(returned), ())) == 1
                  and all(r['correctness'] and r['usage_complete'] for r in runs))
    baseline = medians.get('standard', {}).get('total_tokens')
    ptc = medians.get('ptc', {}).get('total_tokens')
    reduction = 100 * (1 - ptc / baseline) if comparable and baseline and ptc is not None else None
    return dict(medians=medians, token_reduction_percent=reduction,
                note='Medians include failed runs. Unknown usage is null; savings require all runs correct, '
                     'complete usage, equal run counts and matching returned model identifiers.')


def require_smoke(smoke_result, model):
    if not smoke_result.get('correctness') or smoke_result.get('requested_model') != model:
        raise ValueError('Run and inspect a successful hosted-PTC smoke test with this model first.')


def benchmark(client, data, smoke_result, model=DEFAULT_MODEL, repetitions=3, out_dir=None,
              trace=False):
    require_smoke(smoke_result, model)
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError('repetitions must be between 1 and 10')
    path = Path(out_dir or ('artifacts/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    path.mkdir(parents=True, exist_ok=False)
    import openai
    payload = dict(kind='live_api_benchmark', created_at=datetime.now(timezone.utc).isoformat(),
                   sdk_version=openai.__version__, model=model, dataset=data,
                   expected_answer=expected_answer(data), repetitions=repetitions,
                   smoke_result=smoke_result, runs=[], summary={})
    for repetition in range(1, repetitions + 1):
        order = ('standard', 'ptc') if repetition % 2 else ('ptc', 'standard')
        for approach in order:
            result = (run(client, data, approach, model, trace=True) if trace
                      else run(client, data, approach, model))
            result['repetition'] = repetition
            payload['runs'].append(result)
            payload['summary'] = summarize(payload['runs'])
            save_json(payload, path / 'results.json')
            write_csv(payload['runs'], path / 'runs.csv')
            print(f"{repetition} {approach}: correct={result['correctness']} "
                  f"tokens={result['usage']['total_tokens']} error={result['error']}")
            if (result.get('error') or {}).get('status_code') in (400, 401, 403, 404):
                print('Stopping: API rejected the request. Inspect the saved error; no vendor fallback.')
                return payload, path
    return payload, path


def write_csv(runs, path):
    rows = [run_row(r) for r in runs]
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(runs):
    print('mode      rep  correct  input  cached  output  reason  total  requests  tools  seconds')
    for result in runs:
        r = run_row(result)
        values = [r[k] for k in ('approach', 'repetition', 'correctness', *METRICS)]
        print('  '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in values))
    print(json.dumps(summarize(runs), indent=2))


def token_chart(runs, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, result in enumerate(runs):
        tokens = result['usage'].get('total_tokens')
        valid = result['correctness'] and result['usage_complete']
        if tokens is not None:
            ax.bar(i, tokens, color=('#4979a8' if result['approach'] == 'standard' else '#47976b'),
                   hatch='' if valid else '//')
        ax.text(i, tokens or 0, '' if valid else 'failed/unknown', ha='center', fontsize=7)
        ax.set_xticks(range(i + 1), [f"{r['approach']} {r.get('repetition', 1)}" for r in runs[:i+1]])
    ax.set_ylabel('API-reported total tokens')
    ax.set_title('Complete runs; hatched bars mark failed or incomplete measurements')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def load_saved(path):
    payload = json.loads(Path(path).read_text())
    if payload.get('kind') != 'live_api_benchmark' or not payload.get('runs'):
        raise ValueError('Expected a saved live API benchmark, not example data')
    if not all(r.get('responses') for r in payload['runs']):
        raise ValueError('Saved runs lack API responses; inspect the failed audit directly')
    print(f"SAVED RESULTS — NOT A LIVE RUN: {payload['created_at']}")
    return payload


def main():
    parser = argparse.ArgumentParser(description='Explicit travel-budget experiment')
    parser.add_argument('action', choices=['single', 'benchmark', 'show'])
    parser.add_argument('--approach', choices=['standard', 'ptc'], default='standard')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--smoke-artifact', default='artifacts/smoke.json')
    parser.add_argument('--out', help='New benchmark directory, single-run JSON, or saved results JSON')
    args = parser.parse_args()
    if args.action == 'show':
        if not args.out:
            parser.error('show requires --out PATH/results.json')
        payload = load_saved(args.out)
        print_table(payload['runs'])
        token_chart(payload['runs'], Path(args.out).with_name('tokens.png'))
        return
    client, data = make_client(), make_data(small=args.small)
    smoke_result = None
    if args.action == 'benchmark' or args.approach == 'ptc':
        smoke_result = json.loads(Path(args.smoke_artifact).read_text())
        require_smoke(smoke_result, args.model)
    if args.action == 'single':
        result = run(client, data, args.approach, args.model)
        save_json(result, args.out or f'artifacts/{args.approach}-single.json')
        print_table([result])
        raise SystemExit(0 if result['correctness'] else 1)
    payload, path = benchmark(client, data, smoke_result, args.model, args.repetitions, args.out)
    print_table(payload['runs'])
    token_chart(payload['runs'], path / 'tokens.png')
    print(f'Artifacts: {path}')
    if len(payload['runs']) != 2 * args.repetitions or not all(r['correctness'] for r in payload['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
