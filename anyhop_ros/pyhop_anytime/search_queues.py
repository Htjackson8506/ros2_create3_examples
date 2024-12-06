import heapq
from functools import total_ordering


class SearchStack:
    def __init__(self):
        self.stack = []

    def enqueue_all_steps(self, items):
        self.stack.extend(items)

    def dequeue_step(self):
        return self.stack.pop()

    def empty(self):
        return len(self.stack) == 0


@total_ordering
class WrappedPlanStep:
    def __init__(self, step):
        self.step = step

    def __lt__(self, other):
        return self.step.total_cost < other.step.total_cost

    def __eq__(self, other):
        return self.step.total_cost == other.step.total_cost


class HybridQueue:
    def __init__(self):
        self.heap = []
        self.next_pop = None

    def enqueue_all_steps(self, items):
        self.enqueue_all([WrappedPlanStep(item) for item in items])

    def dequeue_step(self):
        d = self.dequeue()
        if d: return d.step

    def empty(self):
        return len(self.heap) == 0 and self.next_pop is None

    def enqueue_all(self, items):
        if self.next_pop:
            assert False
        if len(items) > 0:
            for item in items[:-1]:
                heapq.heappush(self.heap, item)
            self.next_pop = items[-1]

    def dequeue(self):
        if self.next_pop:
            result = self.next_pop
            self.next_pop = None
            return result
        elif self.heap:
            return heapq.heappop(self.heap)


class MonteCarloPlannerHeap:
    def __init__(self, planner, num_samples=10, go_deep_first=True, show_progress=False):
        self.planner = planner
        self.num_samples = num_samples
        self.plan_step_heap = []
        self.go_deep_first = go_deep_first
        self.preferred = None
        self.show_progress = show_progress

    def enqueue_all_steps(self, items):
        rated_steps = []
        for plan_step in items:
            options = self.planner.n_random(plan_step.state, plan_step.start_tasks, self.num_samples)
            if len(options) > 0:
                costs = [plan.total_cost for plan in options]
                rating = sum(costs) / len(costs)
                heapq.heappush(rated_steps, RatedPlanStep(plan_step, rating))

        if self.go_deep_first and len(rated_steps) > 0:
            self.preferred = heapq.heappop(rated_steps)

        for rated_step in rated_steps:
            heapq.heappush(self.plan_step_heap, rated_step)

    def dequeue_step(self):
        if self.preferred is None:
            if self.show_progress:
                popped = heapq.heappop(self.plan_step_heap)
                progress_report("heap", popped)
                return popped.step
            else:
                return heapq.heappop(self.plan_step_heap).step
        else:
            popped = self.preferred
            self.preferred = None
            if self.show_progress:
                progress_report("preferred", popped)
            return popped.step

    def empty(self):
        return self.preferred is None and len(self.plan_step_heap) == 0


def progress_report(origin, popped):
    print(f"From {origin} (depth {popped.step.depth()}) (rating: {popped.rating})\t", end='')
    if popped.step.complete():
        print("Complete!")
    else:
        print("In progress...")


@total_ordering
class RatedPlanStep:
    def __init__(self, step, rating):
        self.step = step
        self.rating = rating

    def __lt__(self, other):
        return self.rating < other.rating

    def __eq__(self, other):
        return self.rating == other.rating
