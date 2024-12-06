import copy
from typing import List, Union, Dict, Set, Callable, Any, Tuple, Sequence, Collection

from pyhop_anytime import TaskList, Planner


def append_text(text: str, seq: List[str]):
    if len(text) > 0:
        seq.append(text)


def tokenize(s: str) -> List[str]:
    result = []
    current = ''
    in_comment = False
    for c in s:
        if in_comment:
            if c == '\n':
                in_comment = False
        elif c == ';':
            in_comment = True
            append_text(current, result)
            current = ''
        elif c == '(':
            append_text(current, result)
            current = ''
            result.append('[')
        elif c == ')':
            append_text(current, result)
            current = ''
            result.append(']')
        elif c.isspace():
            append_text(current, result)
            current = ''
        else:
            current += c
    append_text(current, result)
    return result


def coalesce(tokens: List[str]) -> str:
    result = ''
    for token in tokens:
        if len(result) > 0 and result[-1] != '[' and token != ']':
            result += ', '
        if token in '[]':
            result += token
        else:
            result += f'"{token}"'
    return result


class Parameter:
    def __init__(self, name: str, ptype: str):
        self.name = name
        self.ptype = ptype

    def __repr__(self):
        return f"Parameter('{self.name}', '{self.ptype}')"

    def __eq__(self, other):
        return type(other) == Parameter and other.name == self.name and other.ptype == self.ptype

    def __hash__(self):
        return self.name.__hash__() + self.ptype.__hash__()


class Predicate:
    def __init__(self, name: str, param_list: Sequence[Parameter]):
        self.name = name
        self.param_list = tuple(param_list)

    def __repr__(self):
        return f"Predicate('{self.name}', {self.param_list})"

    def type_list(self) -> List[str]:
        return [p.ptype for p in self.param_list]


def make_pred(pred_list: Sequence) -> Predicate:
    return Predicate(pred_list[0], make_params(pred_list[1:]))


def make_params(param_list: Sequence[str]) -> List[Parameter]:
    result = []
    pending_names = []
    pending_type = False
    for p in param_list:
        if p == '-':
            pending_type = True
        elif pending_type:
            for name in pending_names:
                result.append(Parameter(name, p))
            pending_names = []
            pending_type = False
        else:
            pending_names.append(p)
    return result


def bind_params(params: Sequence[Parameter], args: Sequence[str]) -> Dict[str,str]:
    return {param.name: bound for (param, bound) in zip(params, args)}


class Task:
    def __init__(self, name: str, param_list: Sequence[Parameter]):
        self.name = name
        self.param_list = tuple(param_list)

    def __repr__(self):
        return f"Task('{self.name}', {self.param_list})"

    def type_list(self) -> List[str]:
        return [p.ptype for p in self.param_list]

    def first_task_names(self, domain: 'Domain') -> List[str]:
        result = []
        for method in domain.task2methods[self.name]:
            if len(method.ordered_tasks) > 0:
                result.append(method.ordered_tasks[0].name)
        return result

    def task_func(self, domain: 'Domain') -> Callable[['State', Sequence[str]], TaskList]:
        return lambda state, args: self.task_func_help(domain, state, args)

    def task_func_help(self, domain: 'Domain', state: 'State', args: Sequence[str]) -> TaskList:
        result = []
        for method in domain.task2methods[self.name]:
            bindings = bind_params(method.params, args)
            free_vars = [param for param in method.params if param.name not in bindings]
            if len(free_vars) == 0:
                add_matching_method(domain, method, bindings, state, result)
            else:
                candidate_objects = [state.all_objects_of(free_var.ptype) for free_var in free_vars]
                all_possible = all_combos(candidate_objects)
                for candidate in all_possible:
                    total_bindings = copy.deepcopy(bindings)
                    for i in range(len(free_vars)):
                        total_bindings[free_vars[i].name] = candidate[i]
                    add_matching_method(domain, method, total_bindings, state, result)
        return TaskList(result)


def add_matching_method(domain: 'Domain', method: 'Method', bindings: Dict[str, str], state: 'State',
                        result: List[List[Tuple[str,Sequence[str]]]]):
    passes = True
    if method.precondition is None:
        precond = domain.symbol2preconds[method.name]
        if precond is not None:
            passes = any(p.precondition(bindings, state) for p in precond)
    else:
        passes = method.precondition.precondition(bindings, state)

    if passes:
        result.append([(method.name, tuple([bindings[p.name] for p in method.params]))])


def all_combos(candidates: List[List[Any]]) -> List[List[Any]]:
    if len(candidates) == 0:
        return []
    elif len(candidates) == 1:
        return [[c] for c in candidates[0]]
    else:
        suffixes = all_combos(candidates[1:])
        result = []
        for c in candidates[0]:
            for s in suffixes:
                result.append([c] + s)
        return result


def make_task(task_list) -> Task:
    name = task_list[0]
    assert task_list[1] == ':parameters'
    params = make_params(task_list[2])
    return Task(name, params)


class UntypedSymbol:
    def __init__(self, name: str, positive: bool, param_names: Sequence[str]):
        self.name = name
        self.positive = positive
        self.param_names = tuple(param_names)

    def __repr__(self):
        return f"UntypedSymbol('{self.name}', {self.positive}, {self.param_names})"

    def __eq__(self, other):
        return type(other) == UntypedSymbol and self.name == other.name and self.positive == other.positive and self.param_names == other.param_names

    def __hash__(self):
        return str(self).__hash__()

    def rebind(self, bindings: Dict[str,str]) -> 'UntypedSymbol':
        return UntypedSymbol(self.name, self.positive, [bindings.get(param, param) for param in self.param_names])

    def negated(self) -> 'UntypedSymbol':
        n = copy.deepcopy(self)
        n.positive = not n.positive
        return n

    def precondition(self, bindings: Dict[str,str], state: 'State') -> bool:
        return self.rebind(bindings) in state

    def effect(self, bindings: Dict[str, str], state: 'State'):
        state.add_predicate(self.rebind(bindings))


def make_untyped_symbol(symlist: List) -> UntypedSymbol:
    name = symlist[0]
    args = symlist[1:]
    positive = True
    if name == 'not':
        symlist = symlist[1]
        name = symlist[0]
        args = symlist[1:]
        positive = False
    return UntypedSymbol(name, positive, args)


class State:
    def __init__(self, name: str, objects: List[Parameter], predicates: Collection[UntypedSymbol]):
        self.__name__ = name
        self.predicates = {copy.deepcopy(pred) for pred in predicates if pred.positive}
        self.objects = {obj.name: obj.ptype for obj in objects}
        self.types2objects = {}
        for obj in objects:
            if obj.ptype not in self.types2objects:
                self.types2objects[obj.ptype] = set()
            self.types2objects[obj.ptype].add(obj.name)

    def __repr__(self):
        return f"State('{self.__name__}', {[Parameter(n, t) for (n, t) in self.objects.items()]}, {self.predicates})"

    def __contains__(self, item: Union[str, UntypedSymbol]) -> bool:
        if type(item) == UntypedSymbol:
            present = item in self.predicates
            if not present and not item.positive:
                return item.negated() not in self.predicates
            else:
                return present
        elif type(item) == str:
            return item in self.objects
        else:
            raise TypeError("States only contain objects (str) and predicates (UntypedSymbol)")

    def all_objects_of(self, type_name: str) -> List[str]:
        return [obj for obj in self.types2objects[type_name]]

    def add_predicate(self, predicate: UntypedSymbol):
        if predicate.positive:
            self.predicates.add(predicate)
        else:
            target = copy.deepcopy(predicate)
            target.positive = True
            self.predicates.remove(target)


class Conjunction:
    def __init__(self, predicates: Sequence[UntypedSymbol]):
        self.predicates = tuple(predicates)

    def __repr__(self):
        return f"Conjunction({self.predicates})"

    def rebind(self, bindings: Dict[str, str]) -> 'Conjunction':
        return Conjunction([p.rebind(bindings) for p in self.predicates])

    def precondition(self, bindings: Dict[str, str], state: State) -> bool:
        return all(p.precondition(bindings, state) for p in self.predicates)

    def effect(self, bindings: Dict[str, str], state: State):
        for predicate in self.predicates:
            predicate.effect(bindings, state)


def binding_plus(bindings: Dict[str, str], param: str, value: str) -> Dict[str, str]:
    result = copy.deepcopy(bindings)
    result[param] = value
    return result


class Universal:
    def __init__(self, param: Parameter, pred: UntypedSymbol):
        self.param = param
        self.pred = pred

    def __repr__(self):
        return f"Universal({self.param}, {self.pred})"

    def rebind(self, bindings: Dict[str, str]) -> 'Universal':
        return Universal(self.param, self.pred.rebind(bindings))

    def precondition(self, bindings: Dict[str, str], state: State) -> bool:
        return all(self.pred.precondition(binding_plus(bindings, self.param.name, name), state)
                   for name in state.all_objects_of(self.param.ptype))


def make_precond(prelist: List) -> Union[None, Conjunction, Universal, UntypedSymbol]:
    if len(prelist) == 0:
        return None
    elif prelist[0] == 'forall':
        param = make_params(prelist[1])[0]
        pred = make_untyped_symbol(prelist[2])
        return Universal(param, pred)
    else:
        return make_conjunction(prelist)


def make_conjunction(conjunct_list: List) -> Union[None, Conjunction, UntypedSymbol]:
    if len(conjunct_list) == 0:
        return None
    elif conjunct_list[0] == 'and':
        conjuncts = []
        for conjunct in conjunct_list[1:]:
            conjuncts.append(make_untyped_symbol(conjunct))
        return Conjunction(conjuncts)
    else:
        return make_untyped_symbol(conjunct_list)


class Method:
    def __init__(self, name: str, params: Sequence[Parameter], task_name: str, precondition: Conjunction,
                 ordered_tasks: Sequence[UntypedSymbol]):
        self.name = name
        self.params = tuple(params)
        self.task_name = task_name
        self.precondition = precondition
        self.ordered_tasks = tuple(ordered_tasks)

    def __repr__(self):
        return f"Method('{self.name}', {self.params}, '{self.task_name}', {self.precondition}, {self.ordered_tasks})"

    def first_task_name(self) -> Union[None, str]:
        if len(self.ordered_tasks) > 0:
            return self.ordered_tasks[0].name

    def method_func(self) -> Callable[[State, Sequence[str]], Union[None, TaskList]]:
        return lambda state, args: self.method_func_help(bind_params(self.params, args), state)

    def method_func_help(self, bindings: Dict[str,str], state: State) -> Union[None, TaskList]:
        if self.precondition is None or self.precondition.precondition(bindings, state):
            if len(self.ordered_tasks) == 0:
                return TaskList(completed=True)
            else:
                return TaskList([(task.name, [bindings[param] for param in task.param_names]) for task in self.ordered_tasks])


def make_method(method_list: List) -> Method:
    name = method_list[0]
    params = task_name = preconditions = ordered_tasks = None
    for i in range(1, len(method_list), 2):
        if method_list[i] == ':parameters':
            params = make_params(method_list[i + 1])
        elif method_list[i] == ':task':
            task_name = method_list[i + 1][0]
        elif method_list[i] == ':precondition':
            preconditions = make_precond(method_list[i + 1])
        elif method_list[i] in (':ordered-tasks', ':ordered-subtasks'):
            if method_list[i + 1][0] == 'and':
                ordered_tasks = []
                for task_list in method_list[i + 1][1:]:
                    ordered_tasks.append(make_untyped_symbol(task_list))
            else:
                ordered_tasks = [make_untyped_symbol(method_list[i + 1])]
        else:
            print(f"Unknown tag: {method_list[i]}")
            assert False
    return Method(name, params, task_name, preconditions, ordered_tasks)


class Action:
    def __init__(self, name: str, parameters: Sequence[Parameter],
                 precondition: Union[None, Conjunction, Universal, UntypedSymbol],
                 effects: Union[None, Conjunction, UntypedSymbol]):
        self.name = name
        self.parameters = tuple(parameters)
        self.precondition = precondition
        self.effects = effects

    def __repr__(self):
        return f"Action('{self.name}', {self.parameters}, {self.precondition}, {self.effects})"

    def action_func(self) -> Callable[[State, List[str]], Union[None, State]]:
        return lambda state, args: self.action_func_help(bind_params(self.parameters, args), state)

    def action_func_help(self, bindings: Dict[str,str], state: State) -> Union[None, State]:
        if self.precondition is None or self.precondition.precondition(bindings, state):
            updated = copy.deepcopy(state)
            self.effects.effect(bindings, updated)
            return updated


def make_action(action_list: List) -> Action:
    name = action_list[0]
    parameters = precondition = effects = None
    for i in range(1, len(action_list), 2):
        if action_list[i] == ':parameters':
            parameters = make_params(action_list[i + 1])
        elif action_list[i] == ':precondition':
            precondition = make_precond(action_list[i + 1])
        elif action_list[i] == ':effect':
            effects = make_conjunction(action_list[i + 1])
        else:
            assert False
    return Action(name, parameters, precondition, effects)


class Domain:
    def __init__(self, name: str, types: Set[str], predicates: Dict[str, Predicate], tasks: Dict[str, Task],
                 methods: Dict[str, Method], actions: Dict[str, Action]):
        self.name = name
        self.types = types
        self.predicates = predicates
        self.tasks = tasks
        self.methods = methods
        self.actions = actions

        self.task2methods = {}
        for method in methods.values():
            if method.task_name not in self.task2methods:
                self.task2methods[method.task_name] = []
            self.task2methods[method.task_name].append(method)

        self.symbol2preconds = {}
        unresolved2symbol = {}
        for action_name, action in actions.items():
            self.symbol2preconds[action_name] = [action.precondition]
        for method_name, method in methods.items():
            if method.precondition is None:
                task1 = method.first_task_name()
                if task1 is None:
                    self.symbol2preconds[method_name] = []
                else:
                    if task1 in tasks:
                        unresolved2symbol[method_name] = task1
                    elif task1 in actions:
                        self.symbol2preconds[method_name] = copy.deepcopy(self.symbol2preconds[task1])
                    else:
                        print(f'What is "{task1}" in "{method}"?')
                        assert False
            else:
                self.symbol2preconds[method_name] = [method.precondition]
        for task_name, task in tasks.items():
            self.symbol2preconds[task_name] = []
            for method in self.task2methods[task_name]:
                if method.name in self.symbol2preconds:
                    self.symbol2preconds[task_name].extend(self.symbol2preconds[method.name])
        while len(unresolved2symbol) > 0:
            progress = False
            unresolved = list(unresolved2symbol)
            for candidate in unresolved:
                if unresolved2symbol[candidate] in self.symbol2preconds:
                    precond = self.symbol2preconds[unresolved2symbol[candidate]]
                    self.symbol2preconds[candidate] = copy.deepcopy(precond)
                    self.symbol2preconds[self.methods[candidate].task_name].extend(precond)
                    unresolved2symbol.pop(candidate)
                    progress = True
            assert progress

    def __repr__(self):
        return f"Domain('{self.name}', {self.types}, {self.predicates}, {self.tasks}, {self.methods}, {self.actions})"

    def types_for(self, name: str) -> List[str]:
        if name in self.predicates:
            return self.predicates[name].type_list()
        elif name in self.tasks:
            return self.tasks[name].type_list()
        else:
            print(f"{name} not found")
            assert False

    def make_planner(self) -> Planner:
        planner = Planner()
        for task_name, task in self.tasks.items():
            planner.add_method(task_name, task.task_func(self))
        for method_name, method in self.methods.items():
            planner.add_method(method_name, method.method_func())
        for action_name, action in self.actions.items():
            planner.add_operator(action_name, action.action_func())
        return planner


def parse_domain(name: str, domain_list: List) -> Domain:
    tasks = {}
    methods = {}
    actions = {}
    types = set()
    predicates = {}
    for item in domain_list:
        tag = item[0]
        if tag == ':types':
            types = set(item[1:])
        elif tag == ':predicates':
            predicates = [make_pred(p) for p in item[1:]]
            predicates = {p.name: p for p in predicates}
        elif tag == ':task':
            task = make_task(item[1:])
            tasks[task.name] = task
        elif tag == ':method':
            method = make_method(item[1:])
            methods[method.name] = method
        elif tag == ':action':
            action = make_action(item[1:])
            actions[action.name] = action
    return Domain(name, types, predicates, tasks, methods, actions)


class Problem:
    def __init__(self, name: str, domain: str, objects: List[Parameter], tasks: List[UntypedSymbol],
                 init: List[UntypedSymbol], goal: Conjunction):
        self.name = name
        self.domain = domain
        self.objects = objects
        self.tasks = tasks
        self.init = init
        self.goal = goal

    def __repr__(self):
        return f"Problem('{self.name}', '{self.domain}', {self.objects}, {self.tasks}, {self.init}, {self.goal})"

    def init_state(self) -> State:
        return State(self.name, self.objects, self.init)

    def init_tasks(self) -> List[Tuple[str,Sequence[str]]]:
        return [(task.name, task.param_names) for task in self.tasks]


def parse_problem(name: str, prob_list: List) -> Problem:
    assert prob_list[0][0] == ':domain'
    domain = prob_list[0][1]
    objects = []
    tasks = []
    init = []
    goal = None

    for thing in prob_list[1:]:
        tag = thing[0]
        if tag == ':objects':
            objects = make_params(thing[1:])
        elif tag == ':htn':
            for i in range(1, len(thing), 2):
                if thing[i] in {':subtasks', ':ordered-subtasks', ':ordered-tasks'}:
                    for task in thing[i + 1][1:]:
                        tasks.append(make_untyped_symbol(task[1]))
        elif tag == ':init':
            init = [make_untyped_symbol(s) for s in thing[1:]]
        elif tag == ':goal':
            goal = make_conjunction(thing[1:][0])
        else:
            print(f"Don't recognize tag {tag}")
            assert False
    return Problem(name, domain, objects, tasks, init, goal)


def parse_hddl(hddl_str: str) -> Union[Domain, Problem]:
    py_list_form = eval(coalesce(tokenize(hddl_str)))
    assert py_list_form[0] == 'define'
    category = py_list_form[1][0]
    name = py_list_form[1][1]
    if category == 'domain':
        return parse_domain(name, py_list_form[2:])
    elif category == 'problem':
        return parse_problem(name, py_list_form[2:])
    else:
        assert False


if __name__ == '__main__':
    test_str = open("../../ipc2020-domains/total-order/Robot/domain.hddl").read()
    tokens = tokenize(test_str)
    py_list_form = eval(coalesce(tokens))
    print(py_list_form)

    test_domain = parse_hddl(test_str)
    print(test_domain)

    test_str = open("../../ipc2020-domains/total-order/Blocksworld-HPDDL/pfile_005.hddl").read()
    test_problem = parse_hddl(test_str)
    print(test_problem)