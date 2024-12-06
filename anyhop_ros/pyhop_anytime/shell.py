import os
import readline
import time

import pyhop_anytime
from pyhop_anytime.hddl_parser import parse_hddl

command_list = [
    ("help", "see this message"),
    ("quit/exit", "exit planning shell"),
    ("dir/ls", "list current files"),
    ("cd [dir]", "change directories"),
    ("pwd", "see present working directory"),
    ("hddl [filename]", "Load HDDL problem/domain"),
    ("current", "See current HDDL problem/domain"),
    ("planners", "List all planners"),
    ("see_plan", "Show last plan, along with cost and time of discovery"),
    ("see_plan_stats", "Show last plan cost and time of discovery"),
    ("plan [planner] [time_limit]", "Find a plan using the given planner within the time limit")
]

planners = {
    'dfs': lambda pl, prb, max_s: pl.anyhop(prb.init_state(), prb.init_tasks(), max_s),
    'random': lambda pl, prb, max_s: pl.anyhop_random(prb.init_state(), prb.init_tasks(), max_s),
    'tracked': lambda pl, prb, max_s: pl.anyhop_random_tracked(prb.init_state(), prb.init_tasks(), max_s)
}

if __name__ == '__main__':
    running = True
    domain = None
    problem = None
    planner = None
    last_plan = None
    last_cost = None
    last_discovery = None
    while running:
        cmd = input("> ")
        try:
            if cmd == 'help':
                print("Planning Shell Commands")
                print()
                command_length = max(len(cmd) for cmd, defn in command_list)
                for cmd, defn in command_list:
                    print(f"{cmd}{' ' * (command_length - len(cmd) + 2)}{defn}")
            elif cmd in ('exit', 'quit'):
                running = False
            elif cmd in ('dir', 'ls'):
                for file in sorted(os.listdir()):
                    print(file)
            elif cmd == 'pwd':
                print(os.getcwd())
            elif cmd.startswith("cd"):
                os.chdir(f"{os.getcwd()}/{cmd.split()[1]}")
            elif cmd == 'current':
                print(f"Domain:  {domain.name if domain else 'None'}")
                print(f"Problem: {problem.name if problem else 'None'}")
            elif cmd == 'planners':
                print("Planners")
                print()
                for p in planners:
                    print(p)
            elif cmd.startswith("plan"):
                pname = cmd.split()[1]
                s = float(cmd.split()[2])
                start = time.time()
                result = planners[pname](planner, problem, s)
                duration = time.time() - start
                print(f"{duration}s; {planner.node_expansions} node expansions")
                last_plan, last_cost, last_discovery = result[-1]
            elif cmd.startswith("hddl"):
                parts = cmd.split()
                filename = parts[1]
                s = open(filename).read()
                parsed = parse_hddl(s)
                if type(parsed) == pyhop_anytime.hddl_parser.Domain:
                    domain = parsed
                    planner = domain.make_planner()
                elif type(parsed) == pyhop_anytime.hddl_parser.Problem:
                    problem = parsed
                else:
                    print(f"Unrecognized type: {type(parsed)}")
            elif cmd == "see_plan_stats":
                if last_cost:
                    print(f"Last cost: {last_cost}; discovered after {last_discovery}s")
                else:
                    print("No plan yet found")
            elif cmd == "see_plan":
                for op in last_plan:
                    print(op)
            else:
                print(f"Did not recognize: '{cmd}'")
        except Exception as e:
            print(e)
