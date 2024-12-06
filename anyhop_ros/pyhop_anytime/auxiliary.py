import sys


def anyhop_main(planner):
    if len(sys.argv) == 1:
        print(f"Usage: python3 {sys.argv[0]} -v:[verbosity] -s:[max seconds] [planner_file]+")
    else:
        anyhop_args(planner, sys.argv[1:])


def anyhop_args(planner, args):
    verbosity = find_verbosity(args)
    max_seconds = find_max_seconds(args)
    for filename in args:
        if not filename.startswith("-"):
            exec(open(filename).read())
            plans = planner.anyhop(state, [('start', goals)], max_seconds=max_seconds, verbose=verbosity)
            for (plan, time) in plans:
                print(plan)
            for (plan, time) in plans:
                print(f"Length: {len(plan)} time: {time}")
            print(len(plans), "total plans generated")


def find_verbosity(args) -> int:
    found = find_tag_int(args, "v")
    return 1 if found is None else found


def find_max_seconds(args) -> int:
    return find_tag_int(args, "s")


def find_tag_value(args, tag) -> str:
    for arg in args:
        if arg.startswith(f"-{tag}"):
            return arg.split(":")[1]


def find_tag_int(args, tag) -> int:
    value = find_tag_value(args, tag)
    if value:
        return int(value)