import unittest

from pyhop_anytime.hddl_parser import State, UntypedSymbol, Parameter
from pyhop_anytime.hddl_planner import run_planner


class MyTestCase(unittest.TestCase):
    def test_bug_1(self):
        state = State('pfile_005', [Parameter('b2', 'BLOCK'), Parameter('b5', 'BLOCK'), Parameter('b1', 'BLOCK'),
                                    Parameter('b3', 'BLOCK'), Parameter('b4', 'BLOCK')],
                      {UntypedSymbol('goal_on-table', True, ('b3',)), UntypedSymbol('goal_clear', True, ('b2',)),
                       UntypedSymbol('on-table', True, ('b4',)), UntypedSymbol('done', True, ('b3',)),
                       UntypedSymbol('clear', True, ('b5',)), UntypedSymbol('clear', True, ('b1',)),
                       UntypedSymbol('clear', True, ('b2',)), UntypedSymbol('done', True, ('b4',)),
                       UntypedSymbol('clear', True, ('b4',)), UntypedSymbol('goal_on', True, ('b5', 'b4')),
                       UntypedSymbol('goal_clear', True, ('b1',)), UntypedSymbol('goal_on-table', True, ('b4',)),
                       UntypedSymbol('on-table', True, ('b5',)), UntypedSymbol('done', True, ('b1',)),
                       UntypedSymbol('on-table', True, ('b2',)), UntypedSymbol('goal_on', True, ('b2', 'b5')),
                       UntypedSymbol('on', True, ('b1', 'b3')), UntypedSymbol('hand-empty', True, ()),
                       UntypedSymbol('on-table', True, ('b3',))})
        test_predicate = UntypedSymbol('done', False, ('b1',))
        self.assertFalse(test_predicate.positive)
        self.assertFalse(test_predicate in state.predicates)
        self.assertTrue(test_predicate not in state)

    def _plan_generic(self, domain_name: str, problem_name: str, time_limit: int):
        prefix = '/Users/ferrer/PycharmProjects/ipc2020-domains/total-order'
        domain_file = f"{prefix}/{domain_name}/domain.hddl"
        problem_file = f"{prefix}/{domain_name}/{problem_name}"
        plan_times = run_planner(domain_file, problem_file, time_limit, 0, 'random_tracked')
        for plan, length, duration, state, goals_met in plan_times:
            self.assertTrue(goals_met)

    def test_plan_robot(self):
        self._plan_generic('Robot', 'pfile_01_001.hddl', 3)

    def test_plan_robot_2(self):
        self._plan_generic('Robot', 'pfile_02_002.hddl', 3)

    def test_plan_robot_5(self):
        self._plan_generic('Robot', 'pfile_05_005.hddl', 3)

    def test_plan_blocks(self):
        self._plan_generic('Blocksworld-HPDDL', 'pfile_005.hddl', 3)


if __name__ == '__main__':
    unittest.main()
