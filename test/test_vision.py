from pathlib import Path
from itertools import combinations

import cv2
import numpy as np

from cv_arm_table_setting_demo.layout import load_scene, scene_objects
from cv_arm_table_setting_demo.vision_core import (
    BOUNDING_BOX_COLOUR,
    CLASS_NAMES,
    DISPLAY_NAMES,
    PixelProjector,
    SyntheticShapeKNN,
    _mask_feature,
    segment_observations,
)


CONFIG = Path(__file__).parents[1] / 'config' / 'scene.yaml'


def test_every_class_uses_the_same_debug_box_style_and_readable_name():
    assert BOUNDING_BOX_COLOUR == (40, 220, 40)
    assert set(DISPLAY_NAMES) == set(CLASS_NAMES)
    assert DISPLAY_NAMES == {name: name for name in CLASS_NAMES}


def projector(scene):
    vision = scene['vision']
    camera = vision['camera']
    return PixelProjector(
        camera_x=float(camera['x']),
        camera_y=float(camera['y']),
        camera_z=float(camera['z']),
        plane_z=float(vision['table_plane_z']),
        width=int(camera['width']),
        height=int(camera['height']),
        horizontal_fov=float(camera['horizontal_fov']),
    )


def test_camera_projection_round_trip():
    scene = load_scene(CONFIG)
    model = projector(scene)
    for obj in scene_objects(scene).values():
        u, v = model.world_to_pixel(obj.pose.x, obj.pose.y)
        x, y = model.pixel_to_world(u, v)
        assert abs(x - obj.pose.x) < 1e-9
        assert abs(y - obj.pose.y) < 1e-9
        assert 0 <= u < model.width
        assert 0 <= v < model.height


def test_supervised_classifier_recognizes_every_synthetic_class():
    scene = load_scene(CONFIG)
    model = projector(scene)
    classifier = SyntheticShapeKNN(model.metres_per_pixel)
    features = []
    for name in CLASS_NAMES:
        mask, image = classifier._training_image(name, 1.0, 90.0, 1.0)
        from cv_arm_table_setting_demo.vision_core import _mask_feature
        features.append(_mask_feature(mask, image))
    assignments = classifier.assign_unique(features)
    assert [name for name, _ in assignments] == list(CLASS_NAMES)
    assert all(confidence > 0.70 for _, confidence in assignments)


def test_destination_classifier_recognizes_only_the_placed_subset():
    scene = load_scene(CONFIG)
    model = projector(scene)
    classifier = SyntheticShapeKNN(model.metres_per_pixel)
    names = ('plate', 'cup', 'fork')
    features = []
    for name in names:
        mask, image = classifier._training_image(name, 1.0, 90.0, 1.0)
        from cv_arm_table_setting_demo.vision_core import _mask_feature
        features.append(_mask_feature(mask, image))
    assignments = classifier.assign_subset(features, names)
    assert [name for name, _ in assignments] == list(names)
    assert all(confidence > 0.70 for _, confidence in assignments)


def test_camera_classifier_recognizes_all_1023_object_configurations():
    scene = load_scene(CONFIG)
    model = projector(scene)
    classifier = SyntheticShapeKNN(model.metres_per_pixel)
    features = {}
    for index, name in enumerate(CLASS_NAMES):
        mask, image = classifier._training_image(
            name, 0.97 + 0.006 * index, 13.0 + 19.0 * index, 0.96)
        features[name] = _mask_feature(mask, image)
    checked = 0
    for count in range(1, len(CLASS_NAMES) + 1):
        for selected in combinations(CLASS_NAMES, count):
            assignments = classifier.assign_subset(
                [features[name] for name in selected], selected)
            assert [name for name, _ in assignments] == list(selected)
            assert all(confidence >= 0.50 for _, confidence in assignments)
            checked += 1
    assert checked == 2 ** len(CLASS_NAMES) - 1


def _render_sdf_cutlery_feature(classifier, name, angle):
    size = 160
    centre = size // 2
    mask = np.zeros((size, size), dtype=np.uint8)
    image = np.zeros((size, size, 3), dtype=np.uint8)

    def box(x0, x1, half_width, colour):
        left = centre + int(round(x0 / classifier.metres_per_pixel))
        right = centre + int(round(x1 / classifier.metres_per_pixel))
        width = int(round(half_width / classifier.metres_per_pixel))
        cv2.rectangle(mask, (left, centre - width),
                      (right, centre + width), 255, -1)
        cv2.rectangle(image, (left, centre - width),
                      (right, centre + width), colour, -1)

    if name == 'spoon':
        box(-0.040, 0.090, 0.009, (230, 222, 214))
        axes = (classifier._px(0.035), classifier._px(0.024))
        head = (centre - classifier._px(0.065), centre)
        cv2.ellipse(mask, head, axes, 0, 0, 360, 255, -1)
        cv2.ellipse(image, head, axes, 0, 0, 360, (230, 222, 214), -1)
    elif name == 'fork':
        box(-0.045, 0.095, 0.009, (227, 219, 212))
        box(-0.094, -0.044, 0.0225, (227, 219, 212))
    elif name == 'knife':
        box(-0.100, 0.010, 0.014, (240, 232, 224))
        box(0.010, 0.100, 0.0175, (54, 48, 46))
    else:
        raise KeyError(name)

    matrix = cv2.getRotationMatrix2D((centre, centre), angle, 1.0)
    rotated_mask = cv2.warpAffine(
        mask, matrix, (size, size), flags=cv2.INTER_NEAREST)
    rotated_image = cv2.warpAffine(
        image, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return _mask_feature(rotated_mask, rotated_image)


def test_camera_distinguishes_real_sdf_cutlery_shapes_at_any_orientation():
    scene = load_scene(CONFIG)
    classifier = SyntheticShapeKNN(projector(scene).metres_per_pixel)
    names = ('spoon', 'fork', 'knife')
    for angle in (0.0, 17.0, 43.0, 89.0, 131.0, 177.0):
        features = [
            _render_sdf_cutlery_feature(classifier, name, angle + index * 11.0)
            for index, name in enumerate(names)
        ]
        assignments = classifier.assign_subset(features, names)
        assert [name for name, _ in assignments] == list(names)
        assert all(confidence >= 0.50 for _, confidence in assignments)


def test_complete_rgb_pipeline_uses_pixels_to_recover_object_positions():
    scene = load_scene(CONFIG)
    model = projector(scene)
    classifier = SyntheticShapeKNN(model.metres_per_pixel)
    image = np.full((model.height, model.width, 3), (25, 74, 148), dtype=np.uint8)

    expected_positions = {}
    targets = scene_objects(scene)
    for name in CLASS_NAMES:
        mask, patch = classifier._training_image(name, 1.0, 90.0, 1.0)
        if name in targets:
            pose_x, pose_y = targets[name].pose.x, targets[name].pose.y
        else:
            pose = scene['vision']['distractors'][name]['pose']
            pose_x, pose_y = float(pose['x']), float(pose['y'])
        centre_u, centre_v = model.world_to_pixel(pose_x, pose_y)
        left = int(round(centre_u)) - patch.shape[1] // 2
        top = int(round(centre_v)) - patch.shape[0] // 2
        target = image[top:top + patch.shape[0], left:left + patch.shape[1]]
        target[mask > 0] = patch[mask > 0]
        expected_positions[name] = (pose_x, pose_y)

    observations, mask, _ = segment_observations(
        image=image,
        projector=model,
        bounds=scene['vision']['source_roi'],
        background_distance=float(scene['vision']['background_distance']),
        min_area=float(scene['vision']['min_component_area']),
        expected_count=int(scene['vision']['expected_components']),
    )
    assert len(observations) == 10
    assert cv2.countNonZero(mask) > 0
    assignments = classifier.assign_unique([item.feature for item in observations])
    recovered = {}
    for observation, (name, confidence) in zip(observations, assignments):
        recovered[name] = (*model.pixel_to_world(*observation.centroid), confidence)
    assert set(recovered) == set(CLASS_NAMES)
    for name, (expected_x, expected_y) in expected_positions.items():
        actual_x, actual_y, confidence = recovered[name]
        assert abs(actual_x - expected_x) < 0.035
        assert abs(actual_y - expected_y) < 0.035
        assert confidence > 0.55


def test_destination_rgb_pipeline_recovers_real_clean_table_pick_positions():
    scene = load_scene(CONFIG)
    vision = scene['vision']
    camera = vision['destination_camera']
    model = PixelProjector(
        camera_x=float(camera['x']), camera_y=float(camera['y']),
        camera_z=float(camera['z']), plane_z=float(vision['table_plane_z']),
        width=int(camera['width']), height=int(camera['height']),
        horizontal_fov=float(camera['horizontal_fov']),
    )
    classifier = SyntheticShapeKNN(model.metres_per_pixel)
    image = np.full((model.height, model.width, 3), (64, 15, 184), dtype=np.uint8)
    names = tuple(scene['recipes']['dinner'])
    expected = {}
    for name in names:
        mask, patch = classifier._training_image(name, 1.0, 90.0, 1.0)
        slot = scene['destination_slots'][name]
        x, y = float(slot['x']), float(slot['y'])
        u, v = model.world_to_pixel(x, y)
        left, top = int(round(u)) - 80, int(round(v)) - 80
        target = image[top:top + 160, left:left + 160]
        target[mask > 0] = patch[mask > 0]
        expected[name] = (x, y)
    observations, _, _ = segment_observations(
        image, model, vision['destination_roi'],
        float(vision['background_distance']),
        float(vision['min_component_area']), len(names))
    assignments = classifier.assign_subset(
        [item.feature for item in observations], names)
    recovered = {
        name: model.pixel_to_world(*observation.centroid)
        for observation, (name, _) in zip(observations, assignments)
    }
    assert set(recovered) == set(names)
    for name, (expected_x, expected_y) in expected.items():
        actual_x, actual_y = recovered[name]
        assert abs(actual_x - expected_x) < 0.035
        assert abs(actual_y - expected_y) < 0.035
