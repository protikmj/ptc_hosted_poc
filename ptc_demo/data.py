"""Deterministic synthetic travel data and an independent answer oracle."""
import random


def make_data(small=False, seed=42):
    rng = random.Random(seed)
    count, records = (4, 5) if small else (20, 100)
    budgets = {'L1': 300_000, 'L2': 450_000, 'L3': 600_000}
    names = ['Alex', 'Blair', 'Casey', 'Drew', 'Ellis', 'Frankie', 'Gray', 'Harper',
             'Indigo', 'Jules', 'Kai', 'Lane', 'Morgan', 'Noel', 'Oakley', 'Parker',
             'Quinn', 'Reese', 'Sage', 'Taylor']
    employees, expenses = [], []
    for i in range(count):
        employee_id, level = f'E{i + 1:03}', f'L{i % 3 + 1}'
        employees.append(dict(employee_id=employee_id, name=names[i], level=level,
                              department='Engineering'))
        # Scale record amounts so both workloads retain over/under-budget cases.
        total = budgets[level] * (75 if i % 2 == 0 else 125) // 100 + rng.randint(-1000, 1000)
        weights = [rng.randint(1, 10) for _ in range(records)]
        amounts = [total * w // sum(weights) for w in weights]
        amounts[-1] += total - sum(amounts)
        for j, amount in enumerate(amounts):
            expenses.append(dict(employee_id=employee_id, quarter='Q3',
                                 date=f'2026-{7 + j % 3:02}-{1 + j % 28:02}',
                                 category=rng.choice(['flight', 'hotel', 'meal', 'taxi']),
                                 amount_cents=amount))
    return dict(employees=employees, expenses=expenses, budgets=budgets,
                config=dict(small=small, seed=seed, employee_count=count,
                            records_per_employee=records, currency='USD', year=2026))


def expected_answer(data):
    totals = {}
    for expense in data['expenses']:
        if expense['quarter'] == 'Q3':
            key = expense['employee_id']
            totals[key] = totals.get(key, 0) + expense['amount_cents']
    rows = []
    for employee in data['employees']:
        if employee['department'] != 'Engineering':
            continue
        spent = totals.get(employee['employee_id'], 0)
        budget = data['budgets'][employee['level']]
        if spent > budget:
            rows.append(dict(employee_id=employee['employee_id'], name=employee['name'],
                             spent_cents=spent, budget_cents=budget, over_cents=spent-budget))
    return {'employees': sorted(rows, key=lambda row: row['employee_id'])}


def validate_answer(answer, data):
    if not isinstance(answer, dict) or set(answer) != {'employees'}:
        return False
    rows = answer['employees']
    if not isinstance(rows, list):
        return False
    fields = {'employee_id', 'name', 'spent_cents', 'budget_cents', 'over_cents'}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            return False
        if any(type(row[key]) is not int for key in ('spent_cents', 'budget_cents', 'over_cents')):
            return False
        if any(type(row[key]) is not str for key in ('employee_id', 'name')):
            return False
    return {'employees': sorted(rows, key=lambda row: row['employee_id'])} == expected_answer(data)
