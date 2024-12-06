import random
from typing import Tuple, List, Dict
import queue, math

from pyhop_anytime import Graph

import runner

START_CHAR = 65


class PathwayGrid:
    def __init__(self, meters_per_square: float = 1):
        self.meters_per_square = meters_per_square
        self.visited = set()
        self.blocked = set()
        self.pathway = []

        self.x_min_square = None
        self.x_max_square = None

        self.y_min_square = None
        self.y_max_square = None

        self.rotate = False

    def empty(self) -> bool:
        return len(self.visited) == 0

    def to_squares(self, x_meters: float, y_meters: float) -> Tuple[int, int]:
        return int(x_meters / self.meters_per_square), int(y_meters / self.meters_per_square)

    def to_meters(self, x: int | float, y: int | float) -> Tuple[float, float]:
        return x * self.meters_per_square, y * self.meters_per_square

    def visit(self, x_meters: float, y_meters: float):
        x, y = self.to_squares(x_meters, y_meters)
        self.update_min_max(x, y)
        self.visited.add((x, y))
        self.pathway.append((x, y))

    def ir_check(self, x_meters: float, y_meters: float, orientation: float, ir_value: str):
        orientation = runner.normalize_angle(orientation + runner.IR_HEADINGS[ir_value])

        if -math.pi / 4 < orientation <= math.pi / 4:
            x_meters += self.meters_per_square
        elif math.pi / 4 < orientation <= 3 * math.pi / 4:
            y_meters += self.meters_per_square
        elif 3 * math.pi / 4 < orientation:
            x_meters -= self.meters_per_square
        else:
            y_meters -= self.meters_per_square
        x, y = self.to_squares(x_meters, y_meters)
        self.update_min_max(x, y)
        self.blocked.add((x, y))

    def bump(self, x_meters: float, y_meters: float, orientation: float, bump: str):
        orientation = runner.normalize_angle(orientation + runner.BUMP_HEADINGS[bump])

        if -math.pi / 4 < orientation <= math.pi / 4:
            x_meters += self.meters_per_square
        elif math.pi / 4 < orientation <= 3 * math.pi / 4:
            y_meters += self.meters_per_square
        elif 3 * math.pi / 4 < orientation:
            x_meters -= self.meters_per_square
        else:
            y_meters -= self.meters_per_square
        x, y = self.to_squares(x_meters, y_meters)
        self.update_min_max(x, y)
        self.blocked.add((x, y))

    def update_min_max(self, x: int, y: int):
        if self.empty():
            self.x_min_square = self.x_max_square = x
            self.y_min_square = self.y_max_square = y
        else:
            self.x_min_square = min(self.x_min_square, x)
            self.x_max_square = max(self.x_max_square, x)
            self.y_min_square = min(self.y_min_square, y)
            self.y_max_square = max(self.y_max_square, y)

    def squares_wide(self) -> int:
        return self.x_max_square - self.x_min_square + 1

    def squares_high(self) -> int:
        return self.y_max_square - self.y_min_square + 1

    def console_grid_points(self) -> List[List[Tuple[int, int]]]:
        points = []
        for y in range(self.y_max_square, self.y_min_square - 1, -1):
            row = []
            for x in range(self.x_min_square, self.x_max_square + 1):
                row.append((x, y))
            points.append(row)
        return points

    def inverted_grid_points(self) -> List[List[Tuple[int, int]]]:
        points = []
        for x in range(self.x_min_square, self.x_max_square + 1):
            row = []
            for y in range(self.y_min_square, self.y_max_square + 1):
                row.append((x, y))
            points.append(row)
        return points

    def point2console(self, console_point_list: List[List[Tuple[int, int]]]) -> Dict[Tuple[int,int], Tuple[int, int]]:
        result = {}
        for y, row in enumerate(console_point_list):
            for x, pos in enumerate(row):
                result[pos] = (x, y)
        return result

    def encode_point(self, x: int, y: int) -> str:
        if (x, y) == (0, 0):
            return "$$"

        x_spot = x - self.x_min_square + START_CHAR
        y_spot = y - self.y_min_square + START_CHAR
        return f"{chr(x_spot)}{chr(y_spot)}"

    def decode_point(self, s: str) -> Tuple[int, int]:
        assert len(s) == 2
        if s == '$$':
            return 0, 0
        else:
            return ord(s[0]) - START_CHAR + self.x_min_square, ord(s[1]) - START_CHAR + self.y_min_square

    def occupancy_str(self) -> str:
        if self.empty():
            return "No visits"
        else:
            result = ''
            points = self.inverted_grid_points() if self.rotate else self.console_grid_points()
            for row in points:
                for point in row:
                    if point == (0, 0):
                        result += 'X'
                    else:
                        result += '#' if point in self.blocked else '.' if point in self.visited else ' '
                result += '\n'
            return result

    def path_str(self) -> str:
        if self.empty():
            return "No path"
        else:
            points = self.inverted_grid_points() if self.rotate else self.console_grid_points()
            chars = [[' ' for pos in row] for row in points]
            point2console = self.point2console(points)
            for i in range(len(self.pathway)):
                start = point2console[self.pathway[i]]
                end = point2console[self.pathway[min(len(self.pathway) - 1, i + 1)]]
                chars[start[1]][start[0]] = point_dir_char(start, end)
            rows = [''.join(row) for row in chars]
            return '\n'.join(rows)

    def square_name_str(self) -> str:
        if self.empty():
            return "No map"
        else:
            result = ''
            points = self.inverted_grid_points() if self.rotate else self.console_grid_points()
            for row in points:
                for (x, y) in row:
                    if (x, y) in self.blocked:
                        result += '## '
                    elif (x, y) in self.visited:
                        result += self.encode_point(x, y) + " "
                    else:
                        result += '   '
                result += '\n'
            return result                

    def square_graph(self) -> Graph:
        g = Graph()
        for i, (x, y) in enumerate(self.pathway):
            current_name = self.encode_point(x, y)
            if current_name not in g: 
                g.add_node(current_name, (x * self.meters_per_square, y * self.meters_per_square))
            if i > 0:
                px, py = self.pathway[i - 1]
                prev_name = self.encode_point(px, py)
                g.add_edge(prev_name, current_name)
        return g 

    def centroid_of_unvisited(self) -> Tuple[float, float]:
        num_unvisited = x_total = y_total = 0.0
        for x in range(self.x_min_square, self.x_max_square + 1):
            for y in range(self.y_min_square, self.y_max_square + 1):
                if (x, y) not in self.visited:
                    x_total += x
                    y_total += y
                    num_unvisited += 1
        if num_unvisited > 0:
            return x_total / num_unvisited, y_total / num_unvisited

    def centroid_of_open_space(self, x: float, y: float, max_distance: float) -> Tuple[float, float]:
        x, y = self.to_squares(x, y)
        graph = self.square_graph()
        q = queue.Queue()
        visited = set()
        total_x = total_y = count = 0
        q.put((x, y, 0))
        while not q.empty():
            x, y, distance = q.get()
            if (x, y) not in visited:
                visited.add((x, y))
                total_x += x
                total_y += y
                count += 1
                if distance < max_distance:
                    for neighbor in graph.edges[self.encode_point(x, y)]:
                        nx, ny = self.decode_point(neighbor)
                        q.put((nx, ny, distance + 1))
        return self.to_meters(total_x / count, total_y / count)

    def explore_random_neighbor(self, x_meters: float, y_meters: float) -> Tuple[float, float]:
        x, y = self.to_squares(x_meters, y_meters)
        candidates = [(x + a, y + b) for (a, b) in [(0, 1), (1, 0), (-1, 0), (0, -1)]]
        usable = [(a, b) for (a, b) in candidates if (a, b) not in self.blocked]
        #unvisited = [(a, b) for (a, b) in usable if (a, b) not in self.visited]
        #if len(unvisited) > 0:
        #    a, b = random.choice(unvisited)
        #    return self.to_meters(a, b)
        #elif len(usable) > 0:
        if len(usable) > 0:
            a, b = random.choice(usable)
            return self.to_meters(a, b)

                    
def point_dir_char(start: Tuple[int, int], end: Tuple[int, int]) -> str:
    xdiff = end[0] - start[0]
    ydiff = end[1] - start[1]
    if xdiff == 1:
        if ydiff == 1:
            return '\\'
        elif ydiff == -1:
            return '/'
        else:
            return '>'
    elif xdiff == -1:
        if ydiff == 1:
            return ','
        elif ydiff == -1:
            return '`'
        else:
            return '<'
    else:
        if ydiff == 1:
            return 'v'
        elif ydiff == -1:
            return '^'
        else:
            return '*'
