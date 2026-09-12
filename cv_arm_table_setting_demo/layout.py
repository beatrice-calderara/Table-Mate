"""ROS-independent scene parsing and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


VALID_MEALS = ('breakfast', 'lunch', 'dinner')
CLEAN_COMMAND = 'clean table'
OBJECT_NAMES = (
    'glass', 'wine_glass', 'plate', 'cup', 'spoon', 'fork', 'knife',
    'bottle', 'bowl', 'napkin',
)


@dataclass(frozen=True)
class Pose3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class SceneObject:
    name: str
    label: str
    pose: Pose3
    size: Pose3
    grasp_width: float
    pick_surface_z: float


def load_scene(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    validate_scene(data)
    return data


def validate_scene(data: dict) -> None:
    objects = data.get('objects', {})
    recipes = data.get('recipes', {})
    slots = data.get('destination_slots', {})
    custom_slots = data.get('custom_destination_slots', {})
    if not objects:
        raise ValueError('scene.yaml: objects is empty')
    vision = data.get('vision', {})
    required_vision = {
        'image_topic', 'debug_topic', 'camera', 'source_roi', 'table_plane_z',
        'destination_image_topic', 'destination_debug_topic',
        'destination_camera', 'destination_roi',
        'background_distance', 'min_component_area', 'expected_objects',
        'expected_components', 'stable_frames', 'distractors',
    }
    if not required_vision <= set(vision):
        raise ValueError('scene.yaml: incomplete vision configuration')
    if int(vision['expected_objects']) != len(objects):
        raise ValueError('scene.yaml: expected_objects does not match objects')
    if int(vision['expected_components']) != len(objects) + len(vision['distractors']):
        raise ValueError('scene.yaml: expected_components does not match the scene')
    home = data.get('robot', {}).get('home_joints', {})
    required_home = {
        'base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch',
        'gripper_roll',
    }
    if set(home) != required_home:
        raise ValueError('scene.yaml: incomplete home_joints configuration')
    for meal in VALID_MEALS:
        if meal not in recipes:
            raise ValueError(f'scene.yaml: missing recipe {meal}')
        for name in recipes[meal]:
            if name not in objects:
                raise ValueError(f'{meal}: unknown object {name}')
            if name not in slots:
                raise ValueError(f'{meal}: no destination slot for {name}')
    for name, item in objects.items():
        for key in ('label', 'pose', 'size', 'grasp_width', 'pick_surface_z'):
            if key not in item:
                raise ValueError(f'{name}: missing {key}')
        if item['pick_surface_z'] <= data['tables']['top_z']:
            raise ValueError(f'{name}: pick surface must be above the table')
    selectable = scene_objects(data)
    for name, item in selectable.items():
        if item.pick_surface_z <= float(data['tables']['top_z']):
            raise ValueError(f'{name}: pick surface must be above the table')
    if set(custom_slots) != set(selectable):
        raise ValueError(
            'scene.yaml: custom_destination_slots must contain every object')


def scene_objects(data: dict) -> Dict[str, SceneObject]:
    result: Dict[str, SceneObject] = {}
    selectable = dict(data['objects'])
    selectable.update(data['vision']['distractors'])
    for name, item in selectable.items():
        pose = item['pose']
        size = item['size']
        result[name] = SceneObject(
            name=name,
            label=str(item['label']),
            pose=Pose3(float(pose['x']), float(pose['y']), float(pose['z'])),
            size=Pose3(float(size['x']), float(size['y']), float(size['z'])),
            grasp_width=float(item['grasp_width']),
            pick_surface_z=float(item['pick_surface_z']),
        )
    return result


def recipe(data: dict, meal: str) -> List[str]:
    normalized = meal.strip().lower()
    if normalized not in VALID_MEALS:
        raise ValueError(f'Invalid meal: {meal}. Use: {", ".join(VALID_MEALS)}')
    return list(data['recipes'][normalized])


def normalize_command(command: str) -> str:
    """Normalize user-facing command spelling while keeping a readable API."""
    normalized = ' '.join(command.strip().lower().replace('_', ' ').split())
    if normalized == 'clean':
        return CLEAN_COMMAND
    return normalized


def parse_object_selection(data: dict, command: str) -> List[str]:
    """Validate a comma-separated selection using canonical English names."""
    parts = [
        ' '.join(part.strip().lower().replace('_', ' ').split())
        for part in command.split(',') if part.strip()
    ]
    if not parts:
        raise ValueError('empty object list')
    available = set(scene_objects(data))
    result = []
    for part in parts:
        name = part.replace(' ', '_')
        if name not in OBJECT_NAMES or name not in available:
            raise ValueError(
                f'unknown object "{part}"; use English names: '
                f'{", ".join(OBJECT_NAMES)}')
        if name in result:
            raise ValueError(f'duplicate object: {name}')
        result.append(name)
    return result


def destination(data: dict, object_name: str) -> Pose3:
    slot = data['destination_slots'].get(
        object_name, data['custom_destination_slots'][object_name])
    return Pose3(float(slot['x']), float(slot['y']), float(slot['surface_z']))


def destination_yaw(data: dict, object_name: str) -> float:
    """Requested world yaw for deterministic table-setting orientation."""
    slot = data['destination_slots'].get(
        object_name, data['custom_destination_slots'][object_name])
    return float(slot.get('yaw', 0.0))


def custom_destination(data: dict, object_name: str) -> Pose3:
    slot = data['custom_destination_slots'][object_name]
    return Pose3(float(slot['x']), float(slot['y']), float(slot['surface_z']))


def custom_destination_yaw(data: dict, object_name: str) -> float:
    return float(data['custom_destination_slots'][object_name].get('yaw', 0.0))


def all_names(data: dict) -> Iterable[str]:
    return scene_objects(data).keys()
