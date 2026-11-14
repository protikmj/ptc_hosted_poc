import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from ptc_demo.data import make_data, expected_answer, validate_answer
from ptc_demo.runners import run_loop, sum_usage, smoke
from ptc_demo.tools import TravelTools
from ptc_demo.benchmark import summarize, benchmark, token_chart, load_saved
from ptc_demo.visuals import benchmark_dashboard


class CoreTests(unittest.TestCase):
    def test_dashboard_preserves_failed_runs_and_dataset_switch(self):
        runs = [dict(approach=mode, repetition=1, correctness=correct,
                     requested_model='test', returned_models=['test'], usage_complete=True,
                     usage={'total_tokens': tokens}, elapsed_seconds=seconds,
                     model_requests=3, tool_invocations=8)
                for mode, correct, tokens, seconds in
                [('standard', False, 100, 2), ('ptc', True, 50, 3)]]
        full = dict(runs=runs, model='test', created_at='test timestamp')
        small = copy.deepcopy(full)
        small['runs'][0]['correctness'] = True
        fig = benchmark_dashboard({'Full': full, 'Small': small}, saved=True)
        self.assertEqual(list(fig.data[2].y), [100, 50])
        self.assertEqual(list(fig.data[4].marker.pattern.shape), ['/', ''])
        self.assertIn('SAVED RESULTS', fig.layout.title.text)
        self.assertIn('Savings withheld', fig.layout.title.text)
        buttons = fig.layout.updatemenus[0].buttons
        self.assertEqual(list(buttons[1].args[0]['visible']), [False]*5 + [True]*5)
        self.assertIn('50.00%', buttons[1].args[1]['title']['text'])
        full['runs'][0]['usage_complete'] = False
        unknown = benchmark_dashboard({'Unknown': full}, saved=True)
        self.assertIsNone(unknown.data[2].y[0])
        self.assertIn('usage incomplete', unknown.data[4].customdata[0][0])
        self.assertIn('Plotly.newPlot', fig.to_html(include_plotlyjs=True))

    def test_deterministic_data_and_workload(self):
        data = make_data()
        self.assertEqual(data, make_data())
        self.assertEqual(len(data['employees']), 20)
        self.assertEqual(len(data['expenses']), 2000)
        self.assertTrue(all(type(x['amount_cents']) is int for x in data['expenses']))
        self.assertTrue(3 < len(expected_answer(data)['employees']) < 17)
        self.assertEqual(len(make_data(small=True)['expenses']), 20)

    def test_ground_truth_exact_boundary_and_validation(self):
        data = {'employees': [{'employee_id': 'e1', 'name': 'A', 'level': 'L1',
                               'department': 'Engineering'}],
                'budgets': {'L1': 100}, 'expenses': [
                    {'employee_id': 'e1', 'quarter': 'Q3', 'amount_cents': 100},
                    {'employee_id': 'e1', 'quarter': 'Q2', 'amount_cents': 999}]}
        self.assertEqual(expected_answer(data), {'employees': []})
        data['expenses'][0]['amount_cents'] = 101
        answer = {'employees': [{'employee_id': 'e1', 'name': 'A',
                                'spent_cents': 101, 'budget_cents': 100, 'over_cents': 1}]}
        self.assertEqual(expected_answer(data), answer)
        self.assertTrue(validate_answer(answer, data))
        for field, value in [('employee_id', 'e2'), ('over_cents', 1.0),
                             ('over_cents', True), ('spent_cents', 102)]:
            wrong = copy.deepcopy(answer)
            wrong['employees'][0][field] = value
            self.assertFalse(validate_answer(wrong, data))
        self.assertFalse(validate_answer({'employees': answer['employees'] * 2}, data))

    def test_tools_filter_and_reject_unknown_inputs(self):
        tools = TravelTools(make_data(small=True))
        self.assertEqual(len(tools.get_team_members('Engineering')['employees']), 4)
        self.assertEqual(len(tools.get_expenses('E001', 'Q3')['expenses']), 5)
        with self.assertRaises(ValueError):
            tools.get_expenses('missing', 'Q3')
        with self.assertRaises(ValueError):
            tools.get_budget_by_level('missing')

    def test_usage_subsets_not_added_twice(self):
        usage = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
                 'input_tokens_details': {'cached_tokens': 50},
                 'output_tokens_details': {'reasoning_tokens': 5}}
        self.assertEqual(sum_usage([usage, usage]), {
            'input_tokens': 200, 'cached_input_tokens': 100, 'output_tokens': 40,
            'reasoning_tokens': 10, 'total_tokens': 240})
        self.assertIsNone(sum_usage([{'input_tokens': 1, 'output_tokens': 1,
                                     'total_tokens': 2}])['reasoning_tokens'])

    def test_program_replay_and_caller_preserved(self):
        program = {'type': 'program', 'code': 'generated here', 'call_id': 'p',
                   'fingerprint': 'opaque'}
        caller = {'type': 'program', 'caller_id': 'p'}
        reasoning = {'type': 'reasoning', 'encrypted_content': 'opaque', 'summary': []}
        responses = [
            {'status': 'completed', 'model': 'test', 'output': [reasoning, program,
             {'type': 'function_call', 'name': 'one', 'call_id': 'c',
              'caller': caller, 'arguments': '{}'}]},
            {'status': 'completed', 'model': 'test', 'output': [
             {'type': 'program_output', 'call_id': 'p', 'status': 'completed', 'result': '1'}]},
            {'status': 'completed', 'model': 'test', 'output': [
             {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"value":1}'}]}]}]
        requests = []
        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            raw = responses.pop(0)
            raw['usage'] = {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}
            return SimpleNamespace(model_dump=lambda **_: raw)
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        result = run_loop(client, 'ptc', 'test', 'question', [], {'one': lambda: {'value': 1}})
        self.assertEqual(result['answer'], {'value': 1})
        self.assertEqual(result['model_requests'], 3)
        self.assertEqual(result['tool_invocations'], 1)
        self.assertIn(program, requests[1]['input'])
        self.assertIn(reasoning, requests[1]['input'])
        self.assertEqual(requests[1]['input'][-1]['caller'], caller)
        self.assertEqual(result['usage']['total_tokens'], 36)

    def test_incomplete_usage_retained(self):
        raw = {'status': 'incomplete', 'model': 'test', 'output': [],
               'usage': {'input_tokens': 9, 'output_tokens': 3, 'total_tokens': 12}}
        client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **_: SimpleNamespace(model_dump=lambda **_: raw)))
        result = run_loop(client, 'standard', 'test', 'q', [], {})
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['usage']['total_tokens'], 12)

    def test_failed_run_blocks_savings_and_stays_in_median(self):
        rows = [dict(approach=mode, correctness=True, usage_complete=True,
                     requested_model='test', returned_models=['test'],
                     usage={'total_tokens': tokens}, elapsed_seconds=1,
                     model_requests=1, tool_invocations=1)
                for mode, tokens in [('standard', 100), ('ptc', 50)]]
        self.assertEqual(summarize(rows)['token_reduction_percent'], 50)
        rows[1]['correctness'] = False
        report = summarize(rows)
        self.assertIsNone(report['token_reduction_percent'])
        self.assertEqual(report['medians']['ptc']['total_tokens'], 50)
        self.assertEqual(report['medians']['ptc']['correct_runs'], 0)
        rows[1]['usage_complete'] = False
        self.assertIsNone(summarize(rows)['medians']['ptc']['total_tokens'])

    def test_sdk_smoke_round_trip_without_network(self):
        import httpx2
        from openai import OpenAI
        requests = []
        caller = {'type': 'program', 'caller_id': 'p1'}
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                output = [dict(type='program', id='prog1', call_id='p1',
                               code='/* simulated model program */', fingerprint='opaque'),
                          *[dict(type='function_call', id=f'fc{i}', call_id=f'c{i}',
                                 name=name, arguments='{}', caller=caller)
                            for i, name in enumerate(('get_numbers', 'get_multiplier'))]]
            else:
                output = [dict(type='program_output', call_id='p1', id='po1',
                               status='completed', result='{"value":70}'),
                          dict(type='message', role='assistant', id='msg1', status='completed',
                               content=[dict(type='output_text', text='{"value":70}', annotations=[])])]
            return httpx2.Response(200, json=dict(id=f'resp{len(requests)}', object='response',
                created_at=0, model='test', status='completed', output=output,
                usage=dict(input_tokens=10, output_tokens=5, total_tokens=15)))
        with OpenAI(api_key='offline-test-placeholder', max_retries=0,
                    http_client=httpx2.Client(transport=httpx2.MockTransport(handle))) as client:
            result = smoke(client, 'test')
        self.assertTrue(result['correctness'], result['error'])
        self.assertEqual(result['usage']['total_tokens'], 30)
        self.assertEqual(requests[0]['tools'][-1], {'type': 'programmatic_tool_calling'})
        self.assertEqual(requests[1]['input'][-1]['caller'], caller)
        self.assertEqual(requests[1]['input'][1]['fingerprint'], 'opaque')

    def test_benchmark_order_audit_and_chart(self):
        import matplotlib
        matplotlib.use('Agg')
        orders = []
        def measured_run(client, data, approach, model):
            orders.append(approach)
            return dict(approach=approach, requested_model=model, returned_models=[model],
                        correctness=True, usage_complete=True, error=None, programs=[],
                        responses=[{'id': 'offline-fixture'}],
                        usage=dict(input_tokens=8, cached_input_tokens=0, output_tokens=2,
                                   reasoning_tokens=0, total_tokens=10),
                        model_requests=1, tool_invocations=1, elapsed_seconds=0.1)
        with tempfile.TemporaryDirectory() as directory, patch('ptc_demo.benchmark.run', measured_run):
            payload, path = benchmark(None, make_data(small=True),
                {'correctness': True, 'requested_model': 'test'}, model='test',
                out_dir=Path(directory) / 'audit')
            self.assertEqual(orders, ['standard', 'ptc', 'ptc', 'standard', 'standard', 'ptc'])
            self.assertEqual(len((path / 'runs.csv').read_text().splitlines()), 7)
            self.assertEqual(load_saved(path / 'results.json')['runs'], payload['runs'])
            fig = token_chart(payload['runs'], path / 'tokens.png')
            self.assertGreater((path / 'tokens.png').stat().st_size, 100)
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
