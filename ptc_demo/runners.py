"""Explicit Responses API loop. Generated JavaScript is never executed locally."""
import argparse
import copy
import contextvars
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .data import validate_answer
from .tools import ANSWER_SCHEMA, SCHEMAS, TravelTools, function_schema, object_schema, STRING, INTEGER

DEFAULT_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-6.1-sol')
QUESTION = 'Which employees exceeded their Q3 travel budget, and by how much?'
RULES = '''Use Engineering employees and all their Q3 2026 travel expenses.
All money is integer USD cents. Sum expenses per employee, compare with the
budget for their level, and include only employees whose sum strictly exceeds
the budget. Return employee_id, name, spent_cents, budget_cents, over_cents for
each over-budget employee, sorted by employee_id, inside {"employees": [...]}.
Use tools for all source data. Do not invent records. Do not retry failed tools.'''


def make_client(api_key=None):
    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('No API key: set OPENAI_API_KEY or pass a Databricks secret to make_client().')
    from openai import OpenAI
    from openai.types.responses.tool_param import ProgrammaticToolCalling
    from openai.types.responses.function_tool_param import FunctionToolParam
    if 'allowed_callers' not in FunctionToolParam.__annotations__:
        raise RuntimeError('Install requirements.txt: SDK lacks hosted PTC fields.')
    # Fixed external API endpoint; do not accidentally benchmark a local gateway.
    return OpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=90)


def sum_usage(usages):
    result = {}
    for name, path in {
        'input_tokens': ('input_tokens',), 'output_tokens': ('output_tokens',),
        'total_tokens': ('total_tokens',),
        'cached_input_tokens': ('input_tokens_details', 'cached_tokens'),
        'reasoning_tokens': ('output_tokens_details', 'reasoning_tokens'),
    }.items():
        values = []
        for usage in usages:
            value = usage
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value)
        result[name] = sum(values) if values and all(v is not None for v in values) else None
    return result


def _run_loop(client, approach, model, prompt, schemas, implementations,
              answer_schema=None, max_requests=12, max_output_tokens=8192,
              max_tool_calls=100, trace=False):
    if trace:
        import mlflow

    if approach not in ('standard', 'ptc'):
        raise ValueError('approach must be standard or ptc')
    tools = copy.deepcopy(schemas)
    if approach == 'ptc':
        for tool in tools:
            tool['allowed_callers'] = ['programmatic']
        tools.append({'type': 'programmatic_tool_calling'})
    inputs = [{'role': 'user', 'content': prompt}]
    result = dict(approach=approach, requested_model=model,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  config=dict(max_requests=max_requests, max_output_tokens=max_output_tokens,
                              max_tool_calls=max_tool_calls, reasoning_effort='low', retries=0),
                  prompt=prompt, tools=tools, responses=[], tool_results=[], programs=[],
                  model_requests=0, tool_invocations=0, answer=None, error=None,
                  usage_complete=True, correctness=False)
    start = time.perf_counter()
    try:
        for _ in range(max_requests):
            request = dict(model=model, input=inputs, tools=tools, store=False,
                           include=['reasoning.encrypted_content'],
                           reasoning={'effort': 'low'}, parallel_tool_calls=True,
                           max_output_tokens=max_output_tokens)
            if answer_schema is not None:
                request['text'] = {'format': {'type': 'json_schema', 'name': 'answer',
                                              'strict': True, 'schema': answer_schema}}
            result['model_requests'] += 1
            if trace:
                with mlflow.start_span(
                        name=f"OpenAI Responses request {result['model_requests']}",
                        span_type='CHAT_MODEL') as span:
                    span.set_inputs({'model': model, 'turn': result['model_requests'],
                                     'input_item_count': len(inputs)})
                    response = client.responses.create(**request)
                    raw = response.model_dump(exclude_none=True, by_alias=True)
                    usage = raw.get('usage') or {}
                    token_usage = {key: usage[key] for key in
                                   ('input_tokens', 'output_tokens', 'total_tokens')
                                   if usage.get(key) is not None}
                    span.set_outputs({
                        'model': raw.get('model'),
                        'status': raw.get('status'),
                        'output': [item for item in raw.get('output', [])
                                   if item.get('type') in {
                                       'program', 'program_output', 'function_call', 'message'}],
                        'usage': token_usage,
                    })
                    if token_usage:
                        span.set_attribute('mlflow.chat.tokenUsage', token_usage)
            else:
                response = client.responses.create(**request)
                raw = response.model_dump(exclude_none=True, by_alias=True)
            result['responses'].append(raw)
            if not raw.get('usage'):
                result['usage_complete'] = False
            output = raw.get('output', [])
            result['programs'].extend(item['code'] for item in output if item['type'] == 'program')
            if raw.get('status') != 'completed':
                raise RuntimeError(f"Response status: {raw.get('status')}")
            # Preserve opaque fingerprints, reasoning and every other output item.
            inputs.extend(output)
            calls = [item for item in output if item['type'] == 'function_call']
            if calls:
                if result['tool_invocations'] + len(calls) > max_tool_calls:
                    raise RuntimeError('Tool invocation limit exceeded')
                for call in calls:
                    if call['name'] not in implementations:
                        raise ValueError('Unknown tool name')
                    if approach == 'ptc' and call.get('caller', {}).get('type') != 'program':
                        raise RuntimeError('Expected a hosted program caller')

                def invoke_tool(call):
                    try:
                        args = json.loads(call['arguments'])
                        if not isinstance(args, dict):
                            raise ValueError('Arguments must be an object')
                        value = implementations[call['name']](**args)
                        return value, None
                    except Exception as exc:
                        # Tool errors are visible to both the audit and the model/runtime.
                        return {'error': type(exc).__name__}, type(exc).__name__

                def invoke(call):
                    if not trace:
                        return invoke_tool(call)
                    with mlflow.start_span(name=f"Python tool: {call['name']}",
                                           span_type='TOOL') as span:
                        span.set_inputs({'arguments': call['arguments'],
                                         'caller': call.get('caller'),
                                         'destination': ('hosted runtime' if approach == 'ptc'
                                                         else 'model context')})
                        value, error = invoke_tool(call)
                        span.set_outputs({'result': value, 'error': error})
                        return value, error

                def invoke_in_context(task):
                    context, call = task
                    return context.run(invoke, call)

                result['tool_invocations'] += len(calls)
                with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
                    if trace:
                        tasks = [(contextvars.copy_context(), call) for call in calls]
                        values = list(pool.map(invoke_in_context, tasks))
                    else:
                        values = list(pool.map(invoke, calls))
                for call, (value, error) in zip(calls, values):
                    reply = dict(type='function_call_output', call_id=call['call_id'],
                                 output=json.dumps(value, separators=(',', ':')))
                    if call.get('caller') is not None:
                        reply['caller'] = call['caller']
                    inputs.append(reply)
                    result['tool_results'].append(dict(call=call, reply=reply, error=error,
                        destination='hosted runtime' if approach == 'ptc' else 'model context'))
                continue
            messages = [item for item in output if item['type'] == 'message']
            if messages:
                text = ''.join(part['text'] for item in messages for part in item['content']
                               if part['type'] == 'output_text')
                result['final_text'] = text
                result['answer'] = json.loads(text)
                break
        else:
            raise RuntimeError('Model request limit exceeded')
    except Exception as exc:
        # API exception text may contain request data; persist only safe identifiers.
        result['error'] = {'type': type(exc).__name__,
                           'status_code': getattr(exc, 'status_code', None),
                           'request_id': getattr(exc, 'request_id', None)}
        if type(exc) in (RuntimeError, ValueError):
            result['error']['reason'] = str(exc)
        if len(result['responses']) < result['model_requests']:
            result['usage_complete'] = False
    result['elapsed_seconds'] = time.perf_counter() - start
    result['usage'] = sum_usage([r.get('usage', {}) for r in result['responses']])
    result['returned_models'] = sorted({r['model'] for r in result['responses'] if r.get('model')})
    return result


def run_loop(client, approach, model, prompt, schemas, implementations,
             answer_schema=None, max_requests=12, max_output_tokens=8192,
             max_tool_calls=100, trace=False):
    if not trace:
        return _run_loop(client, approach, model, prompt, schemas, implementations,
                         answer_schema, max_requests, max_output_tokens, max_tool_calls)

    import mlflow

    with mlflow.start_span(name=f'{approach} run', span_type='CHAIN') as span:
        span.set_inputs({'question': prompt, 'model': model, 'approach': approach})
        result = _run_loop(client, approach, model, prompt, schemas, implementations,
                           answer_schema, max_requests, max_output_tokens, max_tool_calls,
                           trace=True)
        span.set_outputs({'final_response': result.get('final_text'),
                          'answer': result.get('answer'), 'error': result.get('error')})
    return result


def run(client, data, approach, model=DEFAULT_MODEL, **limits):
    orchestration = ('Generate orchestration code using hosted PTC. Retrieve, aggregate and filter '
                     'in code. Run independent calls concurrently and emit only the compact answer.'
                     if approach == 'ptc' else
                     'Call the tools directly. Make independent calls in parallel when possible. '
                     'Use the complete tool results to calculate the answer.')
    result = run_loop(client, approach, model, QUESTION + '\n' + RULES + '\n' + orchestration,
                      SCHEMAS, TravelTools(data).implementations(), ANSWER_SCHEMA, **limits)
    result['dataset_config'] = data['config']
    result['correctness'] = (result['error'] is None and validate_answer(result['answer'], data)
                             and not any(t['error'] for t in result['tool_results']))
    if approach == 'ptc':
        result['correctness'] = result['correctness'] and hosted_evidence(result)
    return result


def hosted_evidence(result):
    outputs = [item for response in result['responses'] for item in response.get('output', [])]
    programs = {item['call_id'] for item in outputs if item['type'] == 'program'}
    completed = {item['call_id'] for item in outputs
                 if item['type'] == 'program_output' and item.get('status') == 'completed'}
    callers = {t['call'].get('caller', {}).get('caller_id') for t in result['tool_results']}
    return bool(programs and callers and callers <= programs and programs <= completed)


def smoke(client, model=DEFAULT_MODEL, *, trace=False):
    def get_numbers():
        return {'numbers': [2, 3, 5]}

    def get_multiplier():
        return {'multiplier': 7}

    schemas = [function_schema('get_numbers', 'Return numbers to sum.', {},
                                object_schema({'numbers': {'type': 'array', 'items': INTEGER}})),
               function_schema('get_multiplier', 'Return the multiplier.', {},
                                object_schema({'multiplier': INTEGER}))]
    result = run_loop(client, 'ptc', model,
        'Use a generated hosted program to call get_numbers and get_multiplier, sum the '
        'numbers and multiply by the multiplier. Emit only {"value": integer}. '
        'Return that same JSON as your final answer. Do not retry tools.',
        schemas, dict(get_numbers=get_numbers, get_multiplier=get_multiplier),
        object_schema({'value': INTEGER}), max_requests=6, max_output_tokens=2048,
        max_tool_calls=4, trace=trace)
    result['correctness'] = (result['error'] is None and result['answer'] == {'value': 70}
        and hosted_evidence(result) and result['usage_complete']
        and {t['call']['name'] for t in result['tool_results']} == {'get_numbers', 'get_multiplier'})
    return result


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description='Explicit hosted-PTC smoke test')
    parser.add_argument('action', choices=['smoke'])
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--out', default='artifacts/smoke.json')
    args = parser.parse_args()
    result = smoke(make_client(), args.model)
    save_json(result, args.out)
    print(json.dumps({k: result[k] for k in ('correctness', 'usage', 'error', 'programs')}, indent=2))
    raise SystemExit(0 if result['correctness'] else 1)


if __name__ == '__main__':
    main()
