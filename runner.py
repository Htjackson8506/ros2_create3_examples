import subprocess, time, threading, math, queue, random
import cv2
from typing import Tuple, Iterable

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry
from irobot_create_msgs.msg import WheelStatus, IrIntensityVector, HazardDetectionVector
from irobot_create_msgs.action import RotateAngle, DriveDistance

from fuzzy import *


def drain_queue(q):
    result = None
    while not q.empty():
        result = q.get()
    return result


def straight_twist(vel: float) -> Twist:
    t = Twist()
    t.linear.x = vel
    return t


def turn_twist(vel: float) -> Twist:
    t = Twist()
    t.angular.z = vel
    return t


def turn_twist_towards(vel: float, current_heading: float, goal_heading: float) -> Twist:
    turn_needed = angle_diff(goal_heading, current_heading)
    if turn_needed < 0:
        return turn_twist(-vel)
    else:
        return turn_twist(vel)


def quaternion2euler(orientation: Quaternion) -> Tuple[float, float, float]:
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw)
    roll is rotation around x-axis in radians (counterclockwise)
    pitch is rotation around y-axis in radians (counterclockwise)
    yaw is rotation around z-axis in radians (counterclockwise)
    """
    # Concept: https://www.perplexity.ai/search/explain-the-orientation-jXV4N3xiQUeUGwMTLESZXg

    w, x, y, z = orientation.x, orientation.y, orientation.z, orientation.w

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def discretish_norm(center, radius, num_dice):
    n = sum(random.random() * radius / num_dice for _ in range(num_dice))
    return center + radius / 2 - n


def angle_diff(angle1: float, angle2: float) -> float:
    """
    Find the shortest difference between two angles.
    Parameters should be in radians.
    """
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    diff = normalize_angle(angle1 - angle2)
    return diff if diff <= math.pi else diff - 2 * math.pi


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def euclidean_distance(s1: Iterable[float], s2: Iterable[float]) -> float:
    return math.sqrt(sum((a - b)**2 for (a, b) in zip(s1, s2)))


BUMP_HEADINGS = {
    'bump_left': -math.pi / 2,
    'bump_front_left': -math.pi / 4,
    'bump_front_center': 0.0,
    'bump_front_right': math.pi / 4,
    'bump_right': math.pi / 2
}

IR_HEADINGS = {
    'ir_intensity_side_left': -math.pi / 3,
    'ir_intensity_left': -math.pi / 4,
    'ir_intensity_front_left': -math.pi / 6, 
    'ir_intensity_front_center_left': math.pi,
    'ir_intensity_front_center_right': math.pi,
    'ir_intensity_front_right': math.pi / 6,
    'ir_intensity_right': math.pi / 3
}

HAZARD_HEADINGS = {
    'left': -math.pi / 2,
    'front_left': -math.pi / 4,
    'front_center': 0.0,
    'front_right': math.pi / 4,
    'right': math.pi / 2
}


def find_bump_from(detections):
    for detected in detections:
        if detected.header.frame_id.startswith('bump'):
            return detected.header.frame_id


def find_hazard_from(detections):
    for detected in detections:
        if detected.header.frame_id.startswith('bump') or detected.header.frame_id.startswith('cliff'):
            return detected.header.frame_id


def hazard_id_suffix(frame_id: str) -> str:
    return frame_id[frame_id.find('_') + 1:]


def get_hazard_dir(hazard_frame_id: str) -> float:
    id_suffix = hazard_id_suffix(hazard_frame_id)
    if id_suffix in HAZARD_HEADINGS:
        return HAZARD_HEADINGS[id_suffix]


class Timer:
    def __init__(self):
        self.start = time.time()
        self.count = 0

    def inc(self):
        self.count += 1

    def elapsed(self):
        return self.count / (time.time() - self.start)


class HdxNode(Node):
    def __init__(self, node_name: str, namespace: str = ''):
        super().__init__(node_name)
        if len(namespace) > 0 and not namespace.startswith('/'):
            namespace = f"/{namespace}"
        self.namespace = namespace
        self.start = time.time()
        self.first_callback_time = None
        self.done = False
        self.twist_publisher = self.create_publisher(Twist, f"{namespace}/cmd_vel", 10)
        self.paused = False
        self.child_nodes = []

    def add_self_recursive(self, executor):
        executor.add_node(self)
        for n in self.child_nodes:
            n.add_self_recursive(executor)

    def add_child_nodes(self, *children):
        self.child_nodes.extend(children)

    def pause(self):
        self.paused = True
        for n in self.child_nodes:
            n.pause()

    def resume(self):
        self.paused = False
        for n in self.child_nodes:
            n.resume()

    def subscribe_odom(self, callback):
        self.create_subscription(Odometry, f'{self.namespace}/odom', callback, qos_profile_sensor_data)

    def reset_odom(self):
        call = f'ros2 service call {self.namespace}/reset_pose irobot_create_msgs/srv/ResetPose '
        call += '"{pose:{position:{x: 0.0, y: 0.0, z: 0.0}, orientation:{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"'
        return subprocess.run(call, shell=True, capture_output=True)

    def subscribe_ir(self, callback):
        self.create_subscription(IrIntensityVector, f"{self.namespace}/ir_intensity", callback, qos_profile_sensor_data)

    def subscribe_hazard(self, callback):
        self.create_subscription(HazardDetectionVector, f"{self.namespace}/hazard_detection", callback,
                                 qos_profile_sensor_data)

    def subscribe_wheel(self, callback):
        self.create_subscription(WheelStatus, f'{self.namespace}/wheel_status', callback, qos_profile_sensor_data)

    def publish_twist(self, twist: Twist):
        if not self.paused:
            self.twist_publisher.publish(twist)

    def record_first_callback(self):
        if self.first_callback_time is None:
            self.first_callback_time = self.elapsed_time()

    def ros_issuing_callbacks(self):
        return self.first_callback_time is not None

    def elapsed_time(self):
        return time.time() - self.start

    def quit(self):
        self.done = True

    def reset(self):
        pass

    def quitting(self):
        return self.done


class OdomMonitorNode(HdxNode):
    def __init__(self, name, namespace):
        super().__init__(name, namespace)
        self.subscribe_odom(self.odom_callback)
        self.last_pose = None

    def odom_callback(self, msg: Odometry):
        self.last_pose = msg.pose.pose

    def has_position(self) -> bool:
        return self.last_pose is not None

    def last_x_y(self) -> Tuple[float, float]:
        if self.last_pose is not None:
            p = self.last_pose.position
            return p.x, p.y

    def last_heading(self) -> float:
        return quaternion2euler(self.last_pose.orientation)[0]


class WheelMonitorNode(HdxNode):
    def __init__(self, name, namespace):
        super().__init__(name, namespace)
        self.subscribe_wheel(self.wheel_status_callback)
        self.last_wheel_status = None

    def wheels_stopped(self):
        return self.last_wheel_status is not None and self.last_wheel_status.current_ma_left == 0 and self.last_wheel_status.current_ma_right == 0

    def wheel_status_callback(self, msg):
        self.record_first_callback()
        self.last_wheel_status = msg


class RemoteNode(HdxNode):
    """
    ROS2 node that sends driving messages to the robot based on keyboard input,
    and sends back timing and odometry information.

    Attributes
    ----------
    cmd_queue - receives wasd movement commands
    pos_queue - sends odometry, timestamps, and keys it recognizes
    subscription - receives odometry information
    publisher - sends motor commands

    Methods
    -------
    listener_callback() - sends pose information when received
    timer_callback() - sends timing and key information at time intervals.
    """
    def __init__(self, cmd_queue, pos_queue, ir_queue, bump_queue, namespace: str = ""):
        super().__init__('remote_control_node', namespace)

        self.commands = {
            'w': straight_twist(0.5),
            'a': turn_twist(math.pi/6),
            's': straight_twist(0.0),
            'd': turn_twist(-math.pi/6)
        }

        self.subscribe_odom(self.odom_callback)
        self.subscribe_hazard(self.hazard_callback)
        self.subscribe_ir(self.ir_callback)
        self.create_timer(0.1, self.timer_callback)
        self.cmd_queue = cmd_queue
        self.pos_queue = pos_queue
        self.bump_queue = bump_queue
        self.ir_queue = ir_queue
        self.last_ir = None

    def odom_callback(self, msg: Odometry):
        self.pos_queue.put(msg.pose.pose)

    def hazard_callback(self, msg: HazardDetectionVector):
        b = find_bump_from(msg.detections)
        if b:
            self.bump_queue.put(f"{b} {self.elapsed_time():.2f} {self.last_ir}")

    def ir_callback(self, msg: IrIntensityVector):
        self.last_ir = [reading.value for reading in msg.readings]
        self.ir_queue.put(self.last_ir)

    def timer_callback(self):
        self.pos_queue.put(self.elapsed_time())
        msg = drain_queue(self.cmd_queue)
        if msg is not None and msg in self.commands:
            self.publish_twist(self.commands[msg])
            # Send an echo so the sender knows the publish happened.
            #self.pos_queue.put(msg)


GO_TO_ANGLE_TOLERANCE = math.pi / 32
GO_TO_DISTANCE_TOLERANCE = 0.1


class GoToNode(HdxNode):
    """
    ROS2 node that awaits (x, y) coordinates to which to navigate. The sender is
    responsible for ensuring a clear path from the robot's current location to 
    the specificed position. It sends position data back as it receives it.
    It also maintains a condition variable, which is set when it is active,
    and clear when it is inactive. It becomes active when it receives a 
    command and inactive when it has reached its target destination.
    """
    def __init__(self, pos_queue: queue.Queue, cmd_queue: queue.Queue, status_queue: queue.Queue,
                 active: threading.Event, namespace: str = ""):
        super().__init__('go_to', namespace)
        self.subscribe_odom(self.odom_callback)
        self.pos_queue = pos_queue
        self.cmd_queue = cmd_queue
        self.status_queue = status_queue
        self.active = active
        self.active.clear()
        self.goal_position = None

    def odom_callback(self, pos: Odometry):
        self.pos_queue.put(pos)
        p = pos.pose.pose.position
        h = pos.pose.pose.orientation
        if self.active.is_set():
            self.move_towards_goal(p, h)
        elif self.cmd_queue.empty():
            self.publish_twist(straight_twist(0.0))
            self.status_queue.put("Stopped")
        else:
            self.process_command()

    def move_towards_goal(self, p: Point, h: Quaternion):
        """
        Given a 3D `Point` `p` representing position and a 4D `Quaternion`
        representing orientation, this function creates a `Twist` to move
        the robot towards its goal using `twist_towards_goal()`.
        """
        x, y = self.goal_position
        t = twist_towards_goal(x, y, p, h)
        if t:
            self.publish_twist(t)
            self.status_queue.put(f"linear.x: {t.linear.x:.2f} angular.z: {math.degrees(t.angular.z):.2f}")
        else:
            self.active.clear()
            self.status_queue.put("Stopping")

    def process_command(self):
        msg = self.cmd_queue.get()
        if type(msg) == tuple:
            self.goal_position = msg
            self.active.set()
            self.status_queue.put(f"Received {self.goal_position}")
        elif msg == 'reset':
            self.publish_twist(straight_twist(0.0))
            self.status_queue.put("Resetting odometry...")
            self.reset_odom()
            self.status_queue.put("Reset complete")
        else:
            self.status_queue.put(f"Unrecognized command: {msg}")


def twist_towards_goal(goal_x: float, goal_y: float, p: Point, h: Quaternion,
                       dist_tolerance: float = GO_TO_DISTANCE_TOLERANCE,
                       angle_tolerance: float = GO_TO_ANGLE_TOLERANCE) -> Twist:
    """
    Given a 3D `Point` `p` representing position and a 4D `Quaternion`
    representing orientation, this function creates a `Twist` to move
    the robot towards its goal.

    It creates two fuzzy variables: `far` and `turn`. The basic logic is:
    * if `far` and not `turn`, go forward (defuzzify to `linear.x`)
    * if `turn`, rotate (defuzzify to `angular.z`)

    If the robot is within `angle_tolerance` of the heading towards its
    goal, then `turn` is `False`.

    If the robot is within `dist_tolerance` of its goal, then `far` is
    `False` and it returns `None`.
    """
    euler = quaternion2euler(h)
    angle_disparity = angle_diff(math.atan2(goal_y - p.y, goal_x - p.x), euler[0])
    distance = euclidean_distance((goal_x, goal_y), (p.x, p.y))
    if distance >= dist_tolerance:
        far = fuzzify_rising(distance, 0.0, 0.2)
        turn = fuzzify_rising(abs(angle_disparity), 0.0, angle_tolerance * 4)
        sign = 1 if angle_disparity >= 0 else -1
        t = Twist()
        t.linear.x = defuzzify(min(far, f_not(turn)), 0.0, 0.5)
        t.angular.z = sign * defuzzify(turn, 0.0, math.pi / 4)
        return t


def run_single_node(node_maker):
    finished = threading.Event()
    ros_ready = threading.Event()
    
    st = threading.Thread(target=spin_thread_verbose, args=(finished, ros_ready, node_maker))
    it = threading.Thread(target=input_thread, args=(finished, ros_ready))
    it.start()
    st.start()
    it.join()
    st.join()


class CvKey:
    def __init__(self, key_ascii):
        self.key = chr(key_ascii)

    def is_quit(self):
        return self.key == 'q'


def package_keystroke(keystroke):
    keystroke = keystroke & 0xFF
    if keystroke < 0xFF:
        return CvKey(keystroke)


class OpenCvCode:
    def __init__(self, video_port, frame_proc, frame_proc_args, msg_queue):
        self.msg_queue = msg_queue
        self.video_port = video_port
        self.frame_proc = frame_proc
        self.frame_proc_args = frame_proc_args
        
    def loop(self, finished):
        print("Starting OpenCV loop")
        cap = cv2.VideoCapture(self.video_port)
        timer = Timer()
        while True:
            status, frame = cap.read()
            final_frame, other_data = self.frame_proc(frame, cap, self.frame_proc_args)
            cv2.imshow('OpenCV image', final_frame)
            timer.inc()

            if self.msg_queue.empty():
                self.msg_queue.put(other_data)

            key = package_keystroke(cv2.waitKey(1))
            if key:
                self.msg_queue.put(key)
                if key.is_quit():
                    finished.set()
                    break
        fps = timer.elapsed()
        cap.release()
        cv2.destroyAllWindows()
        print(f"FPS: {fps}")


def run_vision_node(node_maker, cv_object):
    finished = threading.Event()
    ros_ready = threading.Event()

    vt = threading.Thread(target=lambda cv: cv.loop(finished), args=(cv_object,))
    st = threading.Thread(target=spin_thread_verbose, args=(finished, ros_ready, node_maker))
    vt.start()
    st.start()
    vt.join()
    st.join()


def input_thread(finished, ros_ready):
    ros_ready.wait()
    user = input("Type anything to exit")
    finished.set()
    rclpy.shutdown()


def spin_thread(finished, ros_ready, node_maker):
    rclpy.init(args=None)
    executor = rclpy.get_global_executor()
    node = node_maker()
    executor.add_node(node)
    while executor.context.ok() and not finished.is_set() and not node.quitting():
        executor.spin_once()
        if node.ros_issuing_callbacks():
            ros_ready.set()
    node.reset()
    rclpy.shutdown()


def spin_thread_simple(finished, node_maker):
    rclpy.init(args=None)
    executor = rclpy.get_global_executor()
    node = node_maker()
    executor.add_node(node)
    while executor.context.ok() and not finished.is_set() and not node.quitting():
        executor.spin_once()
    node.reset()
    rclpy.shutdown()


def spin_thread_simpler(running, node_maker):
    rclpy.init(args=None)
    executor = rclpy.get_global_executor()
    node = node_maker()
    executor.add_node(node)
    while executor.context.ok() and running.is_set() and not node.quitting():
        executor.spin_once()
    node.reset()
    rclpy.shutdown()


def spin_thread_recursive_node(running, node_maker):
    rclpy.init(args=None)
    executor = rclpy.get_global_executor()
    node = node_maker()
    node.add_self_recursive(executor)
    while executor.context.ok() and running.is_set() and not node.quitting():
        executor.spin_once()
    node.reset()
    rclpy.shutdown()


def spin_thread_verbose(finished, ros_ready, node_maker):
    print("starting")
    rclpy.init(args=None)
    print("init done")
    
    executor = rclpy.get_global_executor()
    node = node_maker()
    executor.add_node(node)
    print("Node created and added")
    while executor.context.ok() and not finished.is_set() and not node.quitting():
        executor.spin_once()
        if node.ros_issuing_callbacks():
            ros_ready.set()
    node.reset()
    rclpy.shutdown()
    print("ROS2 has shut down")


# Inspired by: https://answers.ros.org/question/377848/spinning-multiple-nodes-across-multiple-threads/?answer=377861#post-id-377861
# call rclpy.init() before invoking
def run_multiple_nodes(*nodes):
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    user = input("Type anything to exit")
    rclpy.shutdown()
    executor_thread.join()


# call rclpy.init() before invoking
def run_recursive_node(recursive_node):
    executor = rclpy.executors.MultiThreadedExecutor()
    recursive_node.add_self_recursive(executor)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    user = input("Type anything to exit")
    rclpy.shutdown()
    executor_thread.join()


# call rclpy.init() before invoking
def run_recursive_vision_node(cv_object, recursive_node):
    finished = threading.Event()
    vt = threading.Thread(target=lambda cv: cv.loop(finished), args=(cv_object,))
    vt.start()
    executor = rclpy.executors.MultiThreadedExecutor()
    recursive_node.add_self_recursive(executor)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    vt.join()
    rclpy.shutdown()
    executor_thread.join()


# call rclpy.init() before invoking
def run_vision_multiple_nodes(cv_object, *nodes):
    finished = threading.Event()
    vt = threading.Thread(target=lambda cv: cv.loop(finished), args=(cv_object,))
    vt.start()
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    vt.join()
    rclpy.shutdown()
    executor_thread.join()


class CustomActionClient(Node):
    def __init__(self, name, py_action_type, ros2_action_type, callback, namespace):
        super().__init__(name)
        self._action_client = ActionClient(self, py_action_type, f'{namespace}/{ros2_action_type}')
        self.callback = callback
        self.goal_handle = None

    def send_goal(self, goal_msg):
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if self.goal_handle.accepted:
            print("Goal accepted.")
            self._get_result_future = self.goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.callback)
        else:
            print("Goal rejected...")

    def cancel(self):
        if self.goal_handle is not None:
            future = self.goal_handle.cancel_goal()
            if future:
                print("Goal cancelled")
            else:
                print("Failed to cancel goal")

    def spin_thread(self):
        st = threading.Thread(target=lambda ac: rclpy.spin(ac), args=(self,))
        st.start()


class RotateActionClient(CustomActionClient):
    def __init__(self, callback, namespace):
        super().__init__("RotateActionClient", RotateAngle, 'rotate_angle', callback, namespace)
        
    def send_goal(self, goal_heading, radians_per_sec=1.0):
        goal_msg = RotateAngle.Goal()
        goal_msg.angle = goal_heading
        goal_msg.max_rotation_speed = radians_per_sec
        super().send_goal(goal_msg)


class DriveDistanceClient(CustomActionClient):
    def __init__(self, callback, namespace):
        super().__init__("DriveDistanceClient", DriveDistance, 'drive_distance', callback, namespace)

    def send_goal(self, goal_distance, meters_per_sec=0.3):
        goal_msg = DriveDistance.Goal()
        goal_msg.distance = goal_distance
        goal_msg.max_translation_speed = meters_per_sec
        super().send_goal(goal_msg)
