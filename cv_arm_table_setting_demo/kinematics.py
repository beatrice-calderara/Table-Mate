"""ROS-independent analytic inverse kinematics for the 4-axis arm."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple


def solve_ik(scene: dict, x: float, y: float, palm_z: float) -> Dict[str, float]:
    robot = scene['robot']
    base = robot['base']
    dx, dy = x - float(base['x']), y - float(base['y'])
    radial = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)

    l1 = float(robot['upper_arm_length'])
    l2 = float(robot['forearm_length'])
    hand = float(robot['wrist_to_grip'])
    desired_angle = -math.pi / 2.0
    wrist_r = radial - hand * math.cos(desired_angle)
    wrist_z = palm_z - float(base['shoulder_z']) - hand * math.sin(desired_angle)
    cosine = (wrist_r*wrist_r + wrist_z*wrist_z - l1*l1 - l2*l2) / (2*l1*l2)
    if cosine < -1.0001 or cosine > 1.0001:
        raise ValueError(f'unreachable pose: ({x:.2f}, {y:.2f}, {palm_z:.2f})')
    cosine = max(-1.0, min(1.0, cosine))

    theta2 = -math.acos(cosine)
    theta1 = math.atan2(wrist_z, wrist_r) - math.atan2(
        l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta3 = desired_angle - theta1 - theta2
    result = {
        'base_yaw': yaw,
        'shoulder_pitch': -theta1,
        'elbow_pitch': -theta2,
        'wrist_pitch': -theta3,
    }
    for joint, value in result.items():
        low, high = map(float, robot['joint_limits'][joint])
        if not low <= value <= high:
            raise ValueError(f'{joint}={value:.2f} outside limits [{low}, {high}]')
    return result


def find_reachable_safe_height(
    scene: dict,
    points: Iterable[Tuple[float, float]],
    desired: float,
    minimum: float,
    step: float = 0.01,
) -> float:
    """Find the highest shared travel plane reachable at every XY point."""
    points = tuple(points)
    candidate = float(desired)
    while candidate + 1e-9 >= minimum:
        try:
            for x, y in points:
                solve_ik(scene, x, y, candidate)
            return candidate
        except ValueError:
            candidate -= step
    coordinates = ', '.join(f'({x:.2f}, {y:.2f})' for x, y in points)
    raise ValueError(
        f'no safe reachable travel height for {coordinates}; '
        f'minimum={minimum:.2f}')


def finger_travel_for_grasp(
    grasp_width: float,
    compression_per_finger: float = 0.002,
) -> float:
    """Return symmetric jaw travel for contact or a controlled clamp."""
    return max(
        0.0,
        min(0.105, float(grasp_width) / 2.0 - compression_per_finger),
    )


def gripper_roll_for_yaw(pick_base_yaw: float, place_base_yaw: float,
                         desired_object_yaw: float,
                         initial_object_yaw: float = 0.0) -> float:
    """Compensate base rotation for transfers in either direction."""
    raw = (initial_object_yaw + place_base_yaw - pick_base_yaw
           - desired_object_yaw)
    return math.atan2(math.sin(raw), math.cos(raw))


def gripper_roll_for_object_yaw(base_yaw: float, object_yaw: float) -> float:
    """Align the parallel jaws with an object's yaw at one XY pose.

    The jaw separation axis is undirected, so rolls separated by pi describe
    the same grasp. Returning the nearest equivalent in [-pi/2, pi/2] avoids
    unnecessary wrist rotation while keeping the fingers perpendicular to a
    thin object's long axis.
    """
    raw = float(base_yaw) - float(object_yaw)
    return 0.5 * math.atan2(math.sin(2.0 * raw), math.cos(2.0 * raw))


def gripper_roll_for_directed_object_yaw(
    base_yaw: float,
    object_yaw: float,
) -> float:
    """Align jaws while preserving the directed head-to-handle reference."""
    raw = float(base_yaw) - float(object_yaw)
    return math.atan2(math.sin(raw), math.cos(raw))


def gripper_roll_for_directed_transfer(
    pick_base_yaw: float,
    place_base_yaw: float,
    pick_roll: float,
    initial_object_yaw: float,
    desired_object_yaw: float,
) -> float:
    """Preserve the directed head-to-handle yaw after an oriented grasp."""
    raw = (
        float(pick_roll)
        + float(initial_object_yaw)
        + float(place_base_yaw)
        - float(pick_base_yaw)
        - float(desired_object_yaw)
    )
    return math.atan2(math.sin(raw), math.cos(raw))
