from enum import Enum
import random
import heapq


class Facing(Enum):
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)

    def __neg__(self):
        if self == Facing.NORTH:
            return Facing.SOUTH
        elif self == Facing.SOUTH:
            return Facing.NORTH
        elif self == Facing.EAST:
            return Facing.WEST
        else:
            return Facing.EAST

    def __add__(self, other):
        if type(other) == Facing:
            return self.value[0] + other.value[0], self.value[1] + other.value[1]
        else:
            return self.value[0] + other[0], self.value[1] + other[1]


def facing_from_to(current, successor):
    for f in Facing:
        if f + current == successor:
            return f


def manhattan_distance(p1, p2):
    return sum(abs(p1[i] - p2[i]) for i in range(len(p1)))


def manhattan_neighbors(p):
    return [(p[0] + f.value[0], p[1] + f.value[1]) for f in Facing]


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = set()
        self.dist = {}
        self.prev = {}

    def add_random_obstacles(self, num_obstacles):
        self.obstacles = set()
        obstacle_facings = [Facing.SOUTH, Facing.EAST]
        obstacle_candidates = [(x, y, f) for x in range(self.width - 1) for y in range(self.height - 1)
                               for f in obstacle_facings]
        random.shuffle(obstacle_candidates)
        obstacle_candidate = 0
        while len(self.obstacles) < num_obstacles * 2 and obstacle_candidate < len(obstacle_candidates):
            x1, y1, f = obstacle_candidates[obstacle_candidate]
            self.obstacles.add(((x1, y1), f))
            x2, y2 = f + (x1, y1)
            self.obstacles.add(((x2, y2), -f))
            obstacle_candidate += 1
            if not self.all_cells_reachable():
                self.obstacles.remove(((x1, y1), f))
                self.obstacles.remove(((x2, y2), -f))

    def within(self, p):
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def are_neighbors(self, p1, p2):
        if self.within(p1) and self.within(p2):
            f = facing_from_to(p1, p2)
            if f is not None:
                return (p1, f) not in self.obstacles
        return False

    def all_neighbors(self, p):
        if self.within(p):
            projections = [self.projection(p, f) for f in Facing]
            return [p for p in projections if p is not None]
        else:
            return []

    def projection(self, at, facing):
        if (at, facing) not in self.obstacles:
            future = facing + at
            if self.within(future):
                return future

    def all_cells_reachable(self):
        reached = set()
        heap = [(0, (0, 0))]
        while len(heap) > 0:
            cost, current = heapq.heappop(heap)
            if current not in reached:
                reached.add(current)
                for neighbor in self.all_neighbors(current):
                    heapq.heappush(heap, (1 + cost, neighbor))
        return len(reached) == self.width * self.height

    def print_grid(self, location_char=lambda location: 'O'):
        print(" ", end='')
        for x in range(self.width):
            print(f" {x}", end='')
        print()
        for y in range(self.height):
            if y > 0:
                print("  ", end='')
                for x in range(self.width):
                    if ((x, y), Facing.NORTH) in self.obstacles:
                        print("_ ", end='')
                    else:
                        print(". ", end='')
                print()
            print(f"{y} ", end='')
            for x in range(self.width):
                if x > 0:
                    if ((x, y), Facing.WEST) in self.obstacles:
                        print("|", end='')
                    else:
                        print(".", end='')
                print(location_char((x, y)), end='')
            print()

    def a_star(self, start, end):
        visited = {}
        frontier = [(0, manhattan_distance(start, end), start, None)]
        current = None
        while len(frontier) > 0:
            if current == end:
                path = []
                while current is not None:
                    path = [current] + path
                    current = visited[current]
                return path
            else:
                cost, estimate, current, parent = heapq.heappop(frontier)
                if current not in visited:
                    visited[current] = parent
                    for n in self.all_neighbors(current):
                        heapq.heappush(frontier, (cost + 1, manhattan_distance(n, end), n, current))

    def all_locations(self):
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def shortest_paths_ready(self):
        return len(self.dist) == self.width * self.height

    def floyd_warshall(self):
        # This implementation is derived from Chapter 22 of: https://books.goalkicker.com/AlgorithmsBook/
        vertices = self.all_locations()
        self.dist = {v: {} for v in vertices}
        self.prev = {v: {} for v in vertices}
        for v1 in vertices:
            for v2 in vertices:
                if v1 == v2:
                    self.dist[v1][v1] = 0
                    self.prev[v1][v1] = v1
                elif self.are_neighbors(v1, v2):
                    self.dist[v1][v2] = 1
                    self.prev[v1][v2] = v1
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    i2j = self.dist[i].get(j)
                    i2k = self.dist[i].get(k)
                    k2j = self.dist[k].get(j)
                    if not (i2k is None or k2j is None) and (i2j is None or i2j > i2k + k2j):
                        self.dist[i][j] = i2k + k2j
                        self.prev[i][j] = self.prev[k][j]

    def shortest_path_between(self, p1, p2):
        # This diverges somewhat from Chapter 22 of: https://books.goalkicker.com/AlgorithmsBook/
        #
        # Because the obstacle generator ensures that all locations can be visited, I assume safely
        # that a path exists between any two locations.
        #
        # Because the graph is undirected, I can start from the starting node without any trouble.
        if not self.shortest_paths_ready():
            self.floyd_warshall()
        path = [p1]
        while path[-1] != p2:
            path.append(self.prev[p2][path[-1]])
        return path

    def next_step_from_to(self, current, goal):
        if not self.shortest_paths_ready():
            self.floyd_warshall()
        return self.prev[goal][current]


def show_grid_shortest_paths(grid):
    if not grid.shortest_paths_ready():
        grid.floyd_warshall()
    for p in grid.dist:
        print(p)
        for t in grid.dist[p]:
            print(f"\t{t}: {grid.dist[p][t]} {grid.prev[p][t]}")


if __name__ == '__main__':
    grid = Grid(4, 3)
    grid.add_random_obstacles(8)
    grid.floyd_warshall()
    grid.print_grid()
    show_grid_shortest_paths(grid)
    print(grid.shortest_path_between((0, 0), (grid.width - 1, grid.height - 1)))
    print({grid.next_step_from_to((2, 1), (x, y)) for x in range(grid.width) for y in range(grid.height)})
    print([((x, y), grid.next_step_from_to((2, 1), (x, y))) for x in range(grid.width) for y in range(grid.height)])

