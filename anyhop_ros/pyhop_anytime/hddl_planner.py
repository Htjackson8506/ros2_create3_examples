import sys

from pyhop_anytime.hddl_parser import parse_hddl, Domain, Problem


def run_planner(domain_filename: str, problem_filename: str, max_seconds: float, verbosity: int, planner_name: str):
    with open(domain_filename) as domain_file:
        domain = parse_hddl(domain_file.read())
        assert type(domain) == Domain
        with open(problem_filename) as problem_file:
            problem = parse_hddl(problem_file.read())
            assert type(problem) == Problem
            planner = domain.make_planner()
            planner.print_methods()

            random_tracked = planner_name == 'random_tracked'
            random = planner_name == 'random'
            dfs = planner_name == 'dfs'

            if random_tracked:
                plan_times = planner.anyhop_random_tracked(problem.init_state(), problem.init_tasks(), max_seconds,
                                                           verbose=verbosity)
            if random:
                plan_times = planner.anyhop_random(problem.init_state(), problem.init_tasks(), max_seconds, verbose=verbosity)

            if dfs:
                plan_times = planner.anyhop(problem.init_state(), problem.init_tasks(), max_seconds, verbose=verbosity)

            prelim = [pt + (planner.plan_states(problem.init_state(), pt[0]),) for pt in plan_times]
            result = []
            for plan, length, duration, states in prelim:
                print("goals: ", problem.goal)
                print("state: ", states[-1])
                goals_met = problem.goal.precondition({}, states[-1])
                print("met? ", goals_met)
                result.append((plan, length, duration, states, goals_met))
            return result


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 hddl_planner.py domain.hddl problem.hddl max_seconds [-v=verbosity] [-p=(random_tracked | random | dfs)]")
    else:
        verbosity = 0
        planner_name = 'random_tracked'
        for arg in sys.argv:
            if arg.startswith("-v"):
                verbosity = int(arg.split('=')[1])
            elif arg.startswith("-p"):
                planner_name = arg.split('=')[1]
        plan_times = run_planner(sys.argv[1], sys.argv[2], float(sys.argv[3]), verbosity, planner_name)
        print(planner_name)
        print(plan_times[-1] if len(plan_times) > 0 else "no plan found")