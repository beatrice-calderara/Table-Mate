from pathlib import Path
import math

from cv_arm_table_setting_demo.kinematics import (
    find_reachable_safe_height,
    finger_travel_for_grasp,
    gripper_roll_for_directed_object_yaw,
    gripper_roll_for_directed_transfer,
    gripper_roll_for_object_yaw,
    gripper_roll_for_yaw,
    solve_ik,
)
from cv_arm_table_setting_demo.layout import destination, load_scene, scene_objects


CONFIG = Path(__file__).parents[1] / 'config' / 'scene.yaml'


def test_every_pick_and_place_pose_is_reachable():
    scene = load_scene(CONFIG)
    robot = scene['robot']
    safe_z = float(robot['safe_palm_z'])
    grasp_clearance = float(robot['grasp_clearance'])
    release_clearance = float(robot['release_clearance'])
    for obj in scene_objects(scene).values():
        target = destination(scene, obj.name)
        poses = (
            (obj.pose.x, obj.pose.y, obj.pick_surface_z + grasp_clearance),
            (obj.pose.x, obj.pose.y, safe_z),
            (target.x, target.y, target.z + obj.size.z + grasp_clearance + release_clearance),
            (target.x, target.y, safe_z),
        )
        for pose in poses:
            joints = solve_ik(scene, *pose)
            assert set(joints) == {
                'base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch'
            }


def test_gripper_remains_vertical_for_a_representative_pose():
    scene = load_scene(CONFIG)
    q = solve_ik(scene, -0.68, 0.12, 0.89)
    # In SDF positive pitch rotates local +X toward -Z.
    assert abs(sum(q[name] for name in (
        'shoulder_pitch', 'elbow_pitch', 'wrist_pitch')) - 1.57079632679) < 1e-6


def test_roll_compensation_produces_requested_cutlery_yaw():
    pick_base_yaw = -2.7
    place_base_yaw = -0.6
    desired = 0.0
    roll = gripper_roll_for_yaw(pick_base_yaw, place_base_yaw, desired)
    released_yaw = place_base_yaw - pick_base_yaw - roll
    angular_error = math.atan2(
        math.sin(released_yaw - desired), math.cos(released_yaw - desired))
    assert abs(angular_error) < 1e-6


def test_roll_compensation_is_reversible_for_clean_table():
    table_yaw = 1.5708
    pick_base_yaw = -0.8
    place_base_yaw = -2.2
    desired_source_yaw = 0.0
    roll = gripper_roll_for_yaw(
        pick_base_yaw, place_base_yaw, desired_source_yaw, table_yaw)
    released_yaw = table_yaw + place_base_yaw - pick_base_yaw - roll
    angular_error = math.atan2(
        math.sin(released_yaw - desired_source_yaw),
        math.cos(released_yaw - desired_source_yaw),
    )
    assert abs(angular_error) < 1e-6


def test_jaws_align_with_cutlery_during_destination_pick_and_source_return():
    scene = load_scene(CONFIG)
    safe_z = float(scene['robot']['safe_palm_z'])
    for name in ('fork', 'spoon', 'knife'):
        obj = scene_objects(scene)[name]
        slot = destination(scene, name)
        slot_yaw = float(scene['destination_slots'][name]['yaw'])
        for x, y, object_yaw in (
            (slot.x, slot.y, slot_yaw),
            (obj.pose.x, obj.pose.y, 0.0),
        ):
            base_yaw = solve_ik(scene, x, y, safe_z)['base_yaw']
            roll = gripper_roll_for_object_yaw(base_yaw, object_yaw)
            jaw_axis_yaw = base_yaw + math.pi / 2.0 - roll
            desired_axis_yaw = object_yaw + math.pi / 2.0
            # Jaw axes are equivalent after a half-turn.
            error = 0.5 * math.atan2(
                math.sin(2.0 * (jaw_axis_yaw - desired_axis_yaw)),
                math.cos(2.0 * (jaw_axis_yaw - desired_axis_yaw)),
            )
            assert abs(error) < 1e-9
            assert -math.pi / 2.0 <= roll <= math.pi / 2.0


def test_oriented_cutlery_grasp_preserves_head_handle_direction():
    scene = load_scene(CONFIG)
    safe_z = float(scene['robot']['safe_palm_z'])
    for name in ('fork', 'spoon', 'knife'):
        obj = scene_objects(scene)[name]
        slot = destination(scene, name)
        desired_yaw = float(scene['destination_slots'][name]['yaw'])
        pick_base = solve_ik(
            scene, obj.pose.x, obj.pose.y, safe_z)['base_yaw']
        place_base = solve_ik(scene, slot.x, slot.y, safe_z)['base_yaw']
        pick_roll = gripper_roll_for_object_yaw(pick_base, 0.0)
        place_roll = gripper_roll_for_directed_transfer(
            pick_base, place_base, pick_roll, 0.0, desired_yaw)

        released_yaw = place_base - pick_base - place_roll + pick_roll
        error = math.atan2(
            math.sin(released_yaw - desired_yaw),
            math.cos(released_yaw - desired_yaw),
        )
        assert abs(error) < 1e-9


def test_cutlery_uses_same_absolute_reference_at_pick_and_place():
    scene = load_scene(CONFIG)
    safe_z = float(scene['robot']['safe_palm_z'])
    for name in ('fork', 'spoon', 'knife'):
        obj = scene_objects(scene)[name]
        slot = destination(scene, name)
        desired_yaw = float(scene['destination_slots'][name]['yaw'])
        pick_base = solve_ik(
            scene, obj.pose.x, obj.pose.y, safe_z)['base_yaw']
        place_base = solve_ik(scene, slot.x, slot.y, safe_z)['base_yaw']
        pick_roll = gripper_roll_for_directed_object_yaw(pick_base, 0.0)
        place_roll = gripper_roll_for_directed_object_yaw(
            place_base, desired_yaw)
        pick_reference = pick_base - pick_roll
        place_reference = place_base - place_roll
        error = math.atan2(
            math.sin(place_reference - pick_reference),
            math.cos(place_reference - pick_reference),
        )
        assert abs(error) < 1e-9


def test_every_clean_table_pose_is_reachable():
    scene = load_scene(CONFIG)
    robot = scene['robot']
    safe_z = float(robot['safe_palm_z'])
    grasp_clearance = float(robot['grasp_clearance'])
    release_clearance = float(robot['release_clearance'])
    source_surface = float(scene['tables']['top_z'])
    for obj in scene_objects(scene).values():
        slot = destination(scene, obj.name)
        poses = (
            (slot.x, slot.y, slot.z + obj.size.z + grasp_clearance),
            (slot.x, slot.y, safe_z),
            (obj.pose.x, obj.pose.y,
             source_surface + obj.size.z + grasp_clearance + release_clearance),
            (obj.pose.x, obj.pose.y, safe_z),
        )
        for pose in poses:
            solve_ik(scene, *pose)


def test_cv_offset_near_bottle_uses_a_lower_reachable_travel_plane():
    scene = load_scene(CONFIG)
    safe = find_reachable_safe_height(
        scene,
        points=((-0.87, -0.33), (0.96, -0.18)),
        desired=1.30,
        minimum=1.18,
    )
    assert 1.18 <= safe < 1.30
    solve_ik(scene, -0.87, -0.33, safe)
    solve_ik(scene, 0.96, -0.18, safe)


def test_every_object_is_clamped_between_both_fingers_without_a_gap():
    scene = load_scene(CONFIG)
    for obj in scene_objects(scene).values():
        contact_travel = finger_travel_for_grasp(
            obj.grasp_width, compression_per_finger=0.0)
        travel = finger_travel_for_grasp(obj.grasp_width)
        # With 24 mm jaws centred at +/-12 mm, inner gap = 2 * travel.
        assert abs(2.0 * contact_travel - obj.grasp_width) < 1e-9
        inner_gap = 2.0 * travel
        assert inner_gap < obj.grasp_width
        assert obj.grasp_width - inner_gap <= 0.0041
