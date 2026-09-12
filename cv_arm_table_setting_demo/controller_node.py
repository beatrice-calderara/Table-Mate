"""Articulated-arm table-setting controller with analytic inverse kinematics."""

from __future__ import annotations

import math
import threading
import time
from typing import Dict

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty, Float64, String
from vision_msgs.msg import Detection3DArray

from .layout import (
    CLEAN_COMMAND,
    Pose3,
    VALID_MEALS,
    custom_destination,
    custom_destination_yaw,
    destination,
    destination_yaw,
    load_scene,
    normalize_command,
    parse_object_selection,
    recipe,
    scene_objects,
)
from .kinematics import (
    find_reachable_safe_height,
    finger_travel_for_grasp,
    gripper_roll_for_directed_object_yaw,
    gripper_roll_for_yaw,
    solve_ik,
)


class ArmTableSettingController(Node):
    RATE_HZ = 50.0
    ARM_JOINTS = (
        'base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch',
        'gripper_roll',
    )

    def __init__(self) -> None:
        super().__init__('arm_table_setting_controller')
        default = get_package_share_directory('cv_arm_table_setting_demo') + '/config/scene.yaml'
        self.declare_parameter('scene_config', default)
        self.declare_parameter('meal', 'none')
        self.declare_parameter('speed_scale', 1.0)
        self.scene = load_scene(self.get_parameter('scene_config').value)
        self.objects = scene_objects(self.scene)
        self.speed_scale = max(0.25, min(1.5, float(self.get_parameter('speed_scale').value)))
        self.detected: Dict[str, tuple[float, float, float]] = {}
        self.detection_confidence: Dict[str, float] = {}
        self.destination_detected: Dict[str, tuple[float, float, float]] = {}
        self.destination_detection_confidence: Dict[str, float] = {}
        self.busy = False
        self.placed_objects: list[str] = []
        self.placed_slots: Dict[str, tuple[Pose3, float]] = {}
        self.current_meal: str | None = None
        self.lock = threading.Lock()

        self.joint_pub = {
            name: self.create_publisher(Float64, f'/table_setting/joint/{name}', 10)
            for name in self.ARM_JOINTS
        }
        self.finger_pub = {
            side: self.create_publisher(Float64, f'/table_setting/joint/{side}_finger', 10)
            for side in ('left', 'right')
        }
        self.attach_pub = {
            name: self.create_publisher(Empty, f'/table_setting/grip/{name}/attach', 10)
            for name in self.objects
        }
        self.detach_pub = {
            name: self.create_publisher(Empty, f'/table_setting/grip/{name}/detach', 10)
            for name in self.objects
        }
        self.source_hold_attach_pub = {
            name: self.create_publisher(
                Empty, f'/table_setting/source_hold/{name}/attach', 10)
            for name in self.objects
        }
        self.source_hold_detach_pub = {
            name: self.create_publisher(
                Empty, f'/table_setting/source_hold/{name}/detach', 10)
            for name in self.objects
        }
        self.destination_hold_attach_pub = {
            name: self.create_publisher(
                Empty, f'/table_setting/destination_hold/{name}/attach', 10)
            for name in self.objects
        }
        self.destination_hold_detach_pub = {
            name: self.create_publisher(
                Empty, f'/table_setting/destination_hold/{name}/detach', 10)
            for name in self.objects
        }
        self.status_pub = self.create_publisher(String, '/table_setting/status', 10)
        self.create_subscription(String, '/table_setting/command', self.command_callback, 10)
        self.create_subscription(Detection3DArray, '/table_setting/detections', self.detection_callback, 10)
        self.create_subscription(
            Detection3DArray, '/table_setting/destination_detections',
            self.destination_detection_callback, 10)

        configured_home = self.scene['robot']['home_joints']
        self.home = {
            name: float(configured_home[name]) for name in self.ARM_JOINTS
        }
        self.current = dict(self.home)
        self.finger_position = float(self.scene['robot']['open_finger'])
        self.init_count = 0
        pending = normalize_command(str(self.get_parameter('meal').value))
        self.pending_command = None if pending == 'none' else pending
        self.init_timer = self.create_timer(0.2, self.initialize)
        self.publish_status('STARTING - initializing arm and opening gripper')

    def publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def initialize(self) -> None:
        for name in self.objects:
            self.detach_pub[name].publish(Empty())
            self.destination_hold_detach_pub[name].publish(Empty())
            self.source_hold_attach_pub[name].publish(Empty())
        self.publish_joint_targets(self.home)
        self.publish_fingers(float(self.scene['robot']['open_finger']))
        self.init_count += 1
        # Gazebo is unpaused by the launch after 3 seconds. Continue holding
        # home and detaching until the arm has also had time to settle.
        if self.init_count >= 35:
            self.init_timer.cancel()
            self.publish_status(
                'READY - send breakfast, lunch, dinner, an English list like '
                '"bottle, cup", or clean table')
            if self.pending_command:
                command, self.pending_command = self.pending_command, None
                self.start_job(command)

    def detection_callback(self, msg: Detection3DArray) -> None:
        for detection in msg.detections:
            self.detected[detection.id] = (
                detection.bbox.center.position.x,
                detection.bbox.center.position.y,
                detection.bbox.center.position.z,
            )
            self.detection_confidence[detection.id] = (
                float(detection.results[0].hypothesis.score)
                if detection.results else 0.0)

    def command_callback(self, msg: String) -> None:
        self.start_job(msg.data)

    def destination_detection_callback(self, msg: Detection3DArray) -> None:
        for detection in msg.detections:
            self.destination_detected[detection.id] = (
                detection.bbox.center.position.x,
                detection.bbox.center.position.y,
                detection.bbox.center.position.z,
            )
            self.destination_detection_confidence[detection.id] = (
                float(detection.results[0].hypothesis.score)
                if detection.results else 0.0)

    def start_job(self, command: str) -> None:
        command = normalize_command(command)
        required: list[str] | None = None
        custom = False
        if command not in (*VALID_MEALS, CLEAN_COMMAND):
            try:
                required = parse_object_selection(self.scene, command)
                custom = True
            except ValueError as exc:
                self.publish_status(f'ERROR - {exc}')
                return
        parsed = command if required is None else ', '.join(required)
        self.get_logger().info(
            f'COMMAND_RECEIVED raw="{command}" parsed="{parsed}"')
        with self.lock:
            if self.busy:
                self.publish_status('BUSY - a cycle is already running')
                return
            if command == CLEAN_COMMAND:
                if not self.placed_objects:
                    self.publish_status(
                        'READY - table is already clear; send a meal or object list')
                    return
            elif self.placed_objects:
                self.publish_status(
                    f'ERROR - table is occupied ({self.current_meal}); '
                    f'send {CLEAN_COMMAND} before starting a new meal')
                return
            self.busy = True
        target = self.run_clean_table if command == CLEAN_COMMAND else self.run_job
        args = () if command == CLEAN_COMMAND else (command, required, custom)
        threading.Thread(target=target, args=args, daemon=True).start()

    def run_job(
        self,
        command: str,
        selected: list[str] | None = None,
        custom: bool = False,
    ) -> None:
        try:
            required = selected if custom else recipe(self.scene, command)
            assert required is not None
            # The RGB bridge and the classifier may need a few seconds to
            # deliver three stable frames on a resource-constrained VM.
            deadline = time.monotonic() + 15.0
            # Perception publishes detections only after the complete camera
            # scene has remained stable for the configured number of frames.
            # Do not reject an otherwise valid, uniquely assigned detection
            # because two visually similar objects have a soft score below an
            # arbitrary threshold (for example glass and wine glass).
            while not all(name in self.detected for name in required):
                if time.monotonic() > deadline:
                    unsafe = [
                        name for name in required
                        if name not in self.detected
                    ]
                    raise RuntimeError(
                        'camera recognition missing: '
                        + ', '.join(unsafe))
                time.sleep(0.1)
            job_name = 'custom' if custom else command
            self.current_meal = job_name
            self.destination_detected.clear()
            self.destination_detection_confidence.clear()
            self.publish_status(f'RUNNING {job_name} - objects: {", ".join(required)}')
            for name in required:
                self.pick_and_place(name, custom=custom)
                self.placed_objects.append(name)
            self.move_joints(self.home, duration=3.0)
            self.publish_status(
                f'COMPLETED {job_name} - send {CLEAN_COMMAND} to clear the table')
        except Exception as exc:
            self.get_logger().error(f'Pick-and-place failed: {exc}')
            self.publish_status(
                f'ERROR - {exc}; send {CLEAN_COMMAND} to recover the scene')
            self.move_joints(self.home, duration=3.0)
        finally:
            with self.lock:
                self.busy = False

    def run_clean_table(self) -> None:
        try:
            names = list(reversed(self.placed_objects))
            self.publish_status(
                'WAITING_DESTINATION_VISION - objects: ' + ', '.join(names))
            deadline = time.monotonic() + 20.0
            while not all(name in self.destination_detected for name in names):
                if time.monotonic() > deadline:
                    unsafe = [
                        name for name in names
                        if name not in self.destination_detected
                    ]
                    raise RuntimeError(
                        'destination camera recognition missing: '
                        + ', '.join(unsafe))
                time.sleep(0.1)
            self.publish_status('CLEANING - objects: ' + ', '.join(names))
            for name in names:
                self.return_to_source(name)
                self.placed_objects.remove(name)
                self.placed_slots.pop(name, None)
            self.move_joints(self.home, duration=3.0)
            self.current_meal = None
            self.publish_status(
                'CLEANED - table is clear; send a meal or object list')
        except Exception as exc:
            self.get_logger().error(f'Table clearing failed: {exc}')
            self.publish_status(f'ERROR CLEANING - {exc}')
            self.move_joints(self.home, duration=3.0)
        finally:
            with self.lock:
                self.busy = False

    def pick_and_place(self, name: str, custom: bool = False) -> None:
        obj = self.objects[name]
        # Class identity and XY position both come from the RGB camera.
        px, py, _ = self.detected[name]
        slot = (
            custom_destination(self.scene, name)
            if custom else destination(self.scene, name)
        )
        yaw = (
            custom_destination_yaw(self.scene, name)
            if custom else destination_yaw(self.scene, name)
        )
        self.transfer_object(
            name=name,
            pick_x=px,
            pick_y=py,
            pick_surface_z=obj.pick_surface_z,
            initial_yaw=0.0,
            place_x=slot.x,
            place_y=slot.y,
            place_surface_z=slot.z,
            desired_yaw=yaw,
        )
        self.placed_slots[name] = (slot, yaw)

    def return_to_source(self, name: str) -> None:
        obj = self.objects[name]
        slot, yaw = self.placed_slots[name]
        # Clean-table pickup also uses the destination RGB camera directly.
        measured_x, measured_y, _ = self.destination_detected[name]
        self.transfer_object(
            name=name,
            pick_x=measured_x,
            pick_y=measured_y,
            pick_surface_z=slot.z + obj.size.z,
            initial_yaw=yaw,
            place_x=obj.pose.x,
            place_y=obj.pose.y,
            place_surface_z=float(self.scene['tables']['top_z']),
            desired_yaw=0.0,
            status_prefix='CLEAN_',
        )

    def transfer_object(
        self,
        name: str,
        pick_x: float,
        pick_y: float,
        pick_surface_z: float,
        initial_yaw: float,
        place_x: float,
        place_y: float,
        place_surface_z: float,
        desired_yaw: float,
        status_prefix: str = '',
    ) -> None:
        obj = self.objects[name]
        robot = self.scene['robot']
        # Tall cylinders must be grasped around the body. Using the generic
        # clearance leaves only the finger tips beside the bottle cap and the
        # first contact can topple it.
        grasp_clearance = (
            0.025 if name == 'bottle'
            else float(robot['grasp_clearance'])
        )
        grasp_z = pick_surface_z + grasp_clearance
        release_clearance = (
            0.001 if name in {'fork', 'spoon', 'knife'}
            else float(robot['release_clearance'])
        )
        place_z = (place_surface_z + obj.size.z + grasp_clearance
                   + release_clearance)
        safe_z = self.reachable_safe_height(
            ((pick_x, pick_y), (place_x, place_y)),
            minimum=max(grasp_z, place_z) + 0.06,
        )

        pick_base_yaw = self.inverse_kinematics(
            pick_x, pick_y, safe_z)['base_yaw']
        place_base_yaw = self.inverse_kinematics(
            place_x, place_y, safe_z)['base_yaw']

        if name in {'fork', 'spoon', 'knife'}:
            # Use one absolute, directed reference at both ends. This keeps
            # every utensil straight and prevents a symmetric half-turn from
            # swapping its head and handle.
            pick_roll = gripper_roll_for_directed_object_yaw(
                pick_base_yaw, initial_yaw)
            place_roll = gripper_roll_for_directed_object_yaw(
                place_base_yaw, desired_yaw)
        else:
            # A neutral descent is more reliable for round or broad objects:
            # it avoids catching a glass stem or pushing the plate rim.
            pick_roll = 0.0
            place_roll = gripper_roll_for_yaw(
                pick_base_yaw,
                place_base_yaw,
                desired_yaw,
                initial_yaw,
            )

        self.publish_status(f'{status_prefix}MOVING_TO_PICK {name}')
        self.move_palm(pick_x, pick_y, safe_z, roll=pick_roll)
        self.open_gripper()
        self.move_palm(pick_x, pick_y, grasp_z, roll=pick_roll)

        if name == 'bottle':
            # Secure the bottle before the jaws touch it. The detachable joint
            # prevents a small CV centring error from turning into a fall.
            self.publish_status(f'{status_prefix}ATTACHING {name}')
            self.pulse(self.attach_pub[name], 4, 0.12)
            time.sleep(0.25)
            self.publish_status(f'{status_prefix}CLOSING_GRIPPER {name}')
            self.close_gripper_to_contact(obj.grasp_width)
            time.sleep(0.25)
        else:
            self.publish_status(f'{status_prefix}CLOSING_GRIPPER {name}')
            self.close_gripper_to_contact(obj.grasp_width)
            time.sleep(0.25)
            self.publish_status(f'{status_prefix}ATTACHING {name}')
            self.pulse(self.attach_pub[name], 4, 0.12)
            time.sleep(0.25)
        pick_hold_detach = (
            self.destination_hold_detach_pub[name]
            if status_prefix else self.source_hold_detach_pub[name]
        )
        self.publish_status(f'{status_prefix}RELEASING_TABLE_HOLD {name}')
        self.pulse(pick_hold_detach, 3, 0.10)
        time.sleep(0.15)
        self.publish_status(f'{status_prefix}CLAMPING {name}')
        self.close_gripper_for(obj.grasp_width)
        time.sleep(0.20)

        self.publish_status(f'{status_prefix}LIFTING {name}')
        self.move_palm(pick_x, pick_y, safe_z, roll=pick_roll)
        self.publish_status(f'{status_prefix}MOVING_TO_PLACE {name}')
        self.move_palm(place_x, place_y, safe_z, roll=place_roll)
        self.move_palm(place_x, place_y, place_z, roll=place_roll)

        if name in {'fork', 'spoon', 'knife'}:
            # Repeat the final physical target instead of teleporting the
            # object. This lets the simulated wrist settle fully before the
            # table takes ownership of the straight utensil.
            self.move_palm(
                place_x, place_y, place_z, roll=place_roll)

        self.publish_status(f'{status_prefix}OPENING_GRIPPER {name}')
        self.open_gripper()
        self.publish_status(f'{status_prefix}DETACHING {name}')
        self.pulse(self.detach_pub[name], 4, 0.10)
        place_hold_attach = (
            self.source_hold_attach_pub[name]
            if status_prefix else self.destination_hold_attach_pub[name]
        )
        self.publish_status(f'{status_prefix}LOCKING_ON_TABLE {name}')
        self.pulse(place_hold_attach, 3, 0.10)
        time.sleep(0.35)
        self.move_palm(place_x, place_y, safe_z, roll=place_roll)

    def reachable_safe_height(
        self,
        points: tuple[tuple[float, float], ...],
        minimum: float,
    ) -> float:
        """Lower the travel plane only when CV places an object near reach limits."""
        try:
            return find_reachable_safe_height(
                self.scene,
                points,
                desired=float(self.scene['robot']['safe_palm_z']),
                minimum=minimum,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def inverse_kinematics(self, x: float, y: float, palm_z: float) -> Dict[str, float]:
        try:
            return solve_ik(self.scene, x, y, palm_z)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def move_palm(self, x: float, y: float, z: float,
                  roll: float | None = None) -> None:
        target = self.inverse_kinematics(x, y, z)
        target['gripper_roll'] = (
            self.current['gripper_roll'] if roll is None else self.wrap_angle(roll)
        )
        max_delta = max(abs(target[j] - self.current[j]) for j in self.ARM_JOINTS)
        duration = max(0.8, max_delta / (0.55 * self.speed_scale))
        self.move_joints(target, duration)

    def move_joints(self, target: Dict[str, float], duration: float) -> None:
        start = dict(self.current)
        steps = max(2, int(duration * self.RATE_HZ))
        for step in range(1, steps + 1):
            phase = step / steps
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            values = {
                joint: start[joint] + (target[joint] - start[joint]) * blend
                for joint in self.ARM_JOINTS
            }
            self.publish_joint_targets(values)
            self.current.update(values)
            time.sleep(1.0 / self.RATE_HZ)
        self.current.update(target)
        time.sleep(0.15)

    def publish_joint_targets(self, values: Dict[str, float]) -> None:
        for name, value in values.items():
            self.joint_pub[name].publish(Float64(data=float(value)))

    def publish_fingers(self, value: float) -> None:
        self.finger_position = value
        for publisher in self.finger_pub.values():
            publisher.publish(Float64(data=value))

    def open_gripper(self) -> None:
        self.move_fingers(float(self.scene['robot']['open_finger']))

    def close_gripper_for(self, grasp_width: float) -> None:
        # The 24 mm fingers touch at zero travel. Their inner gap is therefore
        # exactly twice the symmetric joint travel. Compress 2 mm per side so
        # every configured object is visibly centred and clamped by both jaws.
        self.move_fingers(finger_travel_for_grasp(grasp_width))

    def close_gripper_to_contact(self, grasp_width: float) -> None:
        self.move_fingers(finger_travel_for_grasp(
            grasp_width, compression_per_finger=0.0))

    def move_fingers(self, target: float) -> None:
        start = self.finger_position
        for step in range(1, 31):
            blend = 0.5 - 0.5 * math.cos(math.pi * step / 30.0)
            self.publish_fingers(start + (target - start) * blend)
            time.sleep(0.02)

    @staticmethod
    def wrap_angle(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    @staticmethod
    def pulse(publisher, repeats: int, interval: float) -> None:
        for _ in range(repeats):
            publisher.publish(Empty())
            time.sleep(interval)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmTableSettingController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
