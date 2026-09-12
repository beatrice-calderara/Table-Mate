from pathlib import Path
import math

from cv_arm_table_setting_demo.layout import (
    CLEAN_COMMAND,
    VALID_MEALS,
    destination,
    destination_yaw,
    custom_destination,
    custom_destination_yaw,
    load_scene,
    normalize_command,
    parse_object_selection,
    recipe,
    scene_objects,
)


CONFIG = Path(__file__).parents[1] / 'config' / 'scene.yaml'


def test_all_recipes_are_valid():
    scene = load_scene(CONFIG)
    assert set(VALID_MEALS) == set(scene['recipes'])
    assert len(scene['objects']) == 7
    assert set(scene['vision']['distractors']) == {'bottle', 'bowl', 'napkin'}
    assert int(scene['vision']['expected_components']) == 10


def test_requested_meals_match_specification():
    scene = load_scene(CONFIG)
    assert recipe(scene, 'breakfast') == ['plate', 'cup', 'spoon']
    assert recipe(scene, 'lunch') == ['plate', 'fork', 'glass']
    assert recipe(scene, 'dinner') == ['plate', 'fork', 'knife', 'wine_glass']


def test_clean_table_command_accepts_readable_aliases():
    assert normalize_command('clean table') == CLEAN_COMMAND
    assert normalize_command(' CLEAN_TABLE ') == CLEAN_COMMAND
    assert normalize_command('clean') == CLEAN_COMMAND


def test_custom_object_list_accepts_canonical_english_names():
    scene = load_scene(CONFIG)
    assert parse_object_selection(scene, 'bottle, cup') == ['bottle', 'cup']
    assert parse_object_selection(scene, 'plate, wine glass, napkin') == [
        'plate', 'wine_glass', 'napkin']
    assert parse_object_selection(scene, 'fork') == ['fork']


def test_custom_object_list_rejects_unknown_and_duplicate_items():
    scene = load_scene(CONFIG)
    for command in ('lamp', 'cup, cup', 'piatto', ', ,'):
        try:
            parse_object_selection(scene, command)
        except ValueError:
            pass
        else:
            raise AssertionError(f'il comando non valido è stato accettato: {command}')


def test_custom_slots_are_complete_inside_placemat_and_non_overlapping():
    scene = load_scene(CONFIG)
    objects = scene_objects(scene)
    roi = scene['vision']['destination_roi']
    assert set(scene['custom_destination_slots']) == set(objects)
    boxes = {}
    for name, obj in objects.items():
        pose = custom_destination(scene, name)
        yaw = custom_destination_yaw(scene, name)
        quarter_turn = abs(math.sin(yaw)) > 0.7
        width = obj.size.y if quarter_turn else obj.size.x
        depth = obj.size.x if quarter_turn else obj.size.y
        boxes[name] = (
            pose.x - width / 2, pose.x + width / 2,
            pose.y - depth / 2, pose.y + depth / 2,
        )
        assert boxes[name][0] >= roi['x_min']
        assert boxes[name][1] <= roi['x_max']
        assert boxes[name][2] >= roi['y_min']
        assert boxes[name][3] <= roi['y_max']
    names = list(boxes)
    for index, first in enumerate(names):
        ax0, ax1, ay0, ay1 = boxes[first]
        for second in names[index + 1:]:
            bx0, bx1, by0, by1 = boxes[second]
            overlaps = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
            assert not overlaps, f'slot sovrapposti: {first}, {second}'


def test_fixed_slots_follow_a_conventional_place_setting():
    scene = load_scene(CONFIG)
    plate = destination(scene, 'plate')
    glass = destination(scene, 'glass')
    spoon = destination(scene, 'spoon')
    fork = destination(scene, 'fork')
    knife = destination(scene, 'knife')
    wine_glass = destination(scene, 'wine_glass')
    # Diner sits at +X, looking towards -X. Their left is -Y and right is +Y.
    assert fork.y < plate.y < knife.y
    assert spoon.y > plate.y
    assert glass.x < plate.x and glass.y > plate.y
    assert wine_glass.x < plate.x and wine_glass.y > plate.y


def test_cutlery_has_deterministic_parallel_orientation():
    scene = load_scene(CONFIG)
    for name in ('fork', 'knife', 'spoon'):
        assert abs(destination_yaw(scene, name)) < 1e-6


def test_cutlery_heads_point_towards_the_glasses():
    scene = load_scene(CONFIG)
    glass = destination(scene, 'glass')
    for name in ('fork', 'knife', 'spoon'):
        utensil = destination(scene, name)
        yaw = destination_yaw(scene, name)
        # Every head / blade is modelled on local -X.
        head_direction = (-math.cos(yaw), -math.sin(yaw))
        towards_glass = (glass.x - utensil.x, glass.y - utensil.y)
        assert sum(a * b for a, b in zip(head_direction, towards_glass)) > 0.0


def test_all_targets_inside_destination_table():
    scene = load_scene(CONFIG)
    for name in scene['destination_slots']:
        pose = destination(scene, name)
        assert 0.27 <= pose.x <= 1.13
        assert -0.55 <= pose.y <= 0.55


def test_source_layout_has_gripper_safe_clearance_between_every_object():
    scene = load_scene(CONFIG)
    objects = scene_objects(scene)
    roi = scene['vision']['source_roi']
    # Inflate every footprint by 5 cm on all sides. Non-overlap after
    # inflation guarantees at least 10 cm of free space between neighbouring
    # object footprints for the open gripper during clean table.
    clearance = 0.05
    boxes = {}
    for name, obj in objects.items():
        boxes[name] = (
            obj.pose.x - obj.size.x / 2 - clearance,
            obj.pose.x + obj.size.x / 2 + clearance,
            obj.pose.y - obj.size.y / 2 - clearance,
            obj.pose.y + obj.size.y / 2 + clearance,
        )
        assert obj.pose.x - obj.size.x / 2 >= roi['x_min']
        assert obj.pose.x + obj.size.x / 2 <= roi['x_max']
        assert obj.pose.y - obj.size.y / 2 >= roi['y_min']
        assert obj.pose.y + obj.size.y / 2 <= roi['y_max']

    names = list(boxes)
    for index, first in enumerate(names):
        ax0, ax1, ay0, ay1 = boxes[first]
        for second in names[index + 1:]:
            bx0, bx1, by0, by1 = boxes[second]
            overlaps = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
            assert not overlaps, f'unsafe source clearance: {first}, {second}'


def test_home_configuration_is_raised_and_outside_the_tables():
    scene = load_scene(CONFIG)
    robot = scene['robot']
    home = robot['home_joints']
    shoulder = float(home['shoulder_pitch'])
    elbow = float(home['elbow_pitch'])
    wrist = float(home['wrist_pitch'])
    yaw = float(home['base_yaw'])
    total_pitch = shoulder + elbow + wrist
    radial = (
        float(robot['upper_arm_length']) * math.cos(shoulder)
        + float(robot['forearm_length']) * math.cos(shoulder + elbow)
        + float(robot['wrist_to_grip']) * math.cos(total_pitch)
    )
    palm_y = float(robot['base']['y']) + radial * math.sin(yaw)
    palm_z = (
        float(robot['base']['shoulder_z'])
        - float(robot['upper_arm_length']) * math.sin(shoulder)
        - float(robot['forearm_length']) * math.sin(shoulder + elbow)
        - float(robot['wrist_to_grip']) * math.sin(total_pitch)
    )
    assert palm_y > 0.75  # oltre il bordo +Y dei tavoli
    assert palm_z > 1.80
    assert palm_z - 0.14 > 1.65  # anche la punta delle dita resta molto alta
