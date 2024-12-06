import math
import sys, queue, threading, curses, pickle, datetime
import runner
from nav_msgs.msg import Odometry
from irobot_create_msgs.msg import IrIntensityVector

from occupancy_grid import PathwayGrid


class MapperNode(runner.HdxNode):
    def __init__(self, map_str_queue: queue.Queue, namespace: str):
        super().__init__('mapper_node', namespace)
        self.map_str_queue = map_str_queue
        self.subscribe_hazard(self.bump_callback)
        self.subscribe_odom(self.odom_callback)
        self.subscribe_ir(self.ir_callback)
        self.avoid_direction = None
        self.map = PathwayGrid(0.1)
        self.goal = (-1, 0)
        self.last_pose = None
        self.move_start_heading = None

    def assign_goal(self, goal_x, goal_y):
        x, y = self.last_x_y()
        self.move_start_heading = math.atan2(goal_y - y, goal_x - x)
        self.goal = (goal_x, goal_y)

    def bump_clear(self):
        return self.bump is None

    def last_x_y(self):
        p = self.last_pose.position
        return p.x, p.y

    def last_heading(self):
        return runner.quaternion2euler(self.last_pose.orientation)[0]

    def odom_callback(self, pos: Odometry):
        self.last_pose = pos.pose.pose
        x, y = self.last_x_y()
        self.map.visit(x, y)
        twist = None
        if self.goal is not None:
            twist = runner.twist_towards_goal(self.goal[0], self.goal[1], self.last_pose.position,
                                              self.last_pose.orientation)
        if twist:
            self.publish_twist(twist)
        else:
            self.assign_goal(x + math.cos(self.move_start_heading), y + math.sin(self.move_start_heading))

        self.map_str_queue.put((self.goal, self.last_pose, self.map))

    def ir_callback(self, msg: IrIntensityVector):
        self.record_first_callback()
        self.ir_too_close = 400
        ir_values = [reading.value for reading in msg.readings]
        ir_frame_ids = [reading.header.frame_id for reading in msg.readings]
        max_ir = max(ir_values[1:-1]) if self.ir_clear() else max(ir_values)
        if max_ir > self.ir_too_close: # Continue if clear
        #    if self.ir_clear():
        #      mid = len(ir_values) // 2
        #     self.avoid_direction = math.pi / 4
        #    if sum(ir_values[:mid]) > sum(ir_values[-mid:]):
        #       self.avoid_direction *= -1.0
             #self.publish_twist(runner.turn_twist(self.avoid_direction))
        #else:
            self.avoid_direction = None
            x, y = self.last_x_y()
            self.map.ir_check(x, y, self.last_heading(), ir_frame_ids[ir_values.index(max_ir)])
            #print(f'\n {ir_frame_ids[ir_values.index(max_ir)]} \n {max_ir}')
            new_direction = runner.normalize_angle(runner.discretish_norm(self.last_heading() + math.pi, math.pi, 2))
            self.assign_goal(x + math.cos(new_direction), y + math.sin(new_direction))

    def ir_clear(self):
        return self.avoid_direction == None
    
    def is_turning(self):
        return not self.ir_clear()

    def bump_callback(self, msg):
        self.record_first_callback()
        bump = runner.find_bump_from(msg.detections)
        if bump is not None:
            x, y = self.last_x_y()
            self.map.bump(x, y, self.last_heading(), bump)
            new_direction = runner.normalize_angle(runner.discretish_norm(self.last_heading() + math.pi, math.pi, 2))
            self.assign_goal(x + math.cos(new_direction), y + math.sin(new_direction))


class Runner:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.bot = sys.argv[1]
        self.last_map = None
        self.last_pos = None
        self.map_queue = queue.Queue()
        self.running = threading.Event()
        self.robot_thread = threading.Thread(target=runner.spin_thread_simpler,
                                             args=(self.running, lambda: MapperNode(self.map_queue, f"/{self.bot}")))

    def main_loop(self):
        self.running.set()
        self.stdscr.addstr(0, 0, 'Q to quit')
        self.stdscr.refresh()
        self.stdscr.nodelay(True)
        self.robot_thread.start()

        while self.running.is_set():
            try:
                k = self.stdscr.getkey()
                curses.flushinp()
                if k == 'q':
                    self.running.clear()
            except curses.error:
                pass
            self.handle_info()

        self.robot_thread.join()
        with open(f"map_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", 'wb') as file:
            pickle.dump(self.last_map, file)

    def handle_info(self):
        info = runner.drain_queue(self.map_queue)
        if info:
            goal, self.last_pos, self.last_map = info
            graph = self.last_map.square_graph()
            p = self.last_pos.position
            h = self.last_pos.orientation
            name, dist = graph.closest_node(p.x, p.y)
            self.stdscr.addstr(1, 0, f"Position:    {name} ({p.x:6.2f}, {p.y:6.2f}, {p.z:6.2f})       ")
            self.stdscr.addstr(2, 0, f"Goal:        {graph.closest_node(goal[0], goal[1])[0]} ({goal[0]:6.2f}, {goal[1]:6.2f})                      ")
            self.stdscr.addstr(3, 0, f"Orientation: ({h.x:6.2f}, {h.y:6.2f}, {h.z:6.2f}, {h.w:6.2f})        ")
            map_str = self.last_map.square_name_str()
            line_num = 6
            for line in map_str.split('\n'):
                self.stdscr.addstr(line_num, 0, line)
                line_num += 1
            occ_str = self.last_map.occupancy_str()
            line_num += 1
            for line in occ_str.split('\n'):
                self.stdscr.addstr(line_num, 0, line)
                line_num += 1


def run_runner(stdscr):
    r = Runner(stdscr)
    curses.curs_set(0)
    stdscr.clear()
    r.main_loop()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ir_mapper robot_name")
    else:
        curses.wrapper(run_runner)
