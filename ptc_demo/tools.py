"""The same three Python implementations serve both orchestration modes."""


def object_schema(properties):
    return dict(type='object', properties=properties, required=list(properties),
                additionalProperties=False)


def function_schema(name, description, parameters, output):
    return dict(type='function', name=name, description=description, strict=True,
                parameters=object_schema(parameters), output_schema=output)


STRING = {'type': 'string'}
INTEGER = {'type': 'integer'}
EMPLOYEE = object_schema(dict(employee_id=STRING, name=STRING, level=STRING, department=STRING))
EXPENSE = object_schema(dict(employee_id=STRING, quarter=STRING, date=STRING,
                             category=STRING, amount_cents=INTEGER))
ANSWER_SCHEMA = object_schema({'employees': {'type': 'array', 'items': object_schema(
    dict(employee_id=STRING, name=STRING, spent_cents=INTEGER, budget_cents=INTEGER,
         over_cents=INTEGER))}})


class TravelTools:
    def __init__(self, data):
        self.data = data

    def get_team_members(self, department):
        if department != 'Engineering':
            raise ValueError('Unknown department')
        return {'employees': [e.copy() for e in self.data['employees'] if e['department'] == department]}

    def get_expenses(self, employee_id, quarter):
        if employee_id not in {e['employee_id'] for e in self.data['employees']} or quarter != 'Q3':
            raise ValueError('Unknown employee or unsupported quarter')
        return {'expenses': [e.copy() for e in self.data['expenses']
                             if e['employee_id'] == employee_id and e['quarter'] == quarter]}

    def get_budget_by_level(self, level):
        if level not in self.data['budgets']:
            raise ValueError('Unknown level')
        return {'level': level, 'budget_cents': self.data['budgets'][level]}

    def implementations(self):
        return {name: getattr(self, name) for name in
                ('get_team_members', 'get_expenses', 'get_budget_by_level')}


SCHEMAS = [
    function_schema('get_team_members', 'Get employees in a department.',
                    {'department': STRING}, object_schema({'employees': {'type': 'array', 'items': EMPLOYEE}})),
    function_schema('get_expenses', 'Get all travel expenses for one employee and quarter in 2026, in USD cents.',
                    dict(employee_id=STRING, quarter=STRING),
                    object_schema({'expenses': {'type': 'array', 'items': EXPENSE}})),
    function_schema('get_budget_by_level', 'Get the Q3 travel budget in USD cents for one employee level.',
                    {'level': STRING}, object_schema(dict(level=STRING, budget_cents=INTEGER))),
]
