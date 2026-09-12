"""Image segmentation, synthetic training and supervised object classification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


TARGET_CLASS_NAMES = (
    'glass', 'wine_glass', 'plate', 'cup', 'spoon', 'fork', 'knife')
ADDITIONAL_CLASS_NAMES = ('bottle', 'bowl', 'napkin')
CLASS_NAMES = TARGET_CLASS_NAMES + ADDITIONAL_CLASS_NAMES

# Uniform object-debug style shared by every recognized class.
BOUNDING_BOX_COLOUR = (40, 220, 40)
DISPLAY_NAMES = {
    name: name for name in CLASS_NAMES
}


@dataclass(frozen=True)
class PixelProjector:
    """Pinhole projection for a fixed camera looking vertically down."""

    camera_x: float
    camera_y: float
    camera_z: float
    plane_z: float
    width: int
    height: int
    horizontal_fov: float

    @property
    def metres_per_pixel(self) -> float:
        distance = self.camera_z - self.plane_z
        return 2.0 * distance * math.tan(self.horizontal_fov / 2.0) / self.width

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        scale = self.metres_per_pixel
        # Gazebo camera: forward +X, image right -Y, image down -Z. After the
        # +90 degree pitch, image down corresponds to world -X.
        x = self.camera_x - (v - self.height / 2.0) * scale
        y = self.camera_y - (u - self.width / 2.0) * scale
        return x, y

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        scale = self.metres_per_pixel
        u = self.width / 2.0 - (y - self.camera_y) / scale
        v = self.height / 2.0 - (x - self.camera_x) / scale
        return u, v


@dataclass
class VisualObservation:
    contour: np.ndarray
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    feature: np.ndarray


def _mask_feature(mask: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Build a scale-aware descriptor from silhouette, geometry and colour."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError('empty visual mask')
    x, y, width, height = cv2.boundingRect(points)
    rectangle = cv2.minAreaRect(points)
    rect_width, rect_height = rectangle[1]
    raw_aspect = max(rect_width, rect_height) / max(
        1.0, min(rect_width, rect_height))
    crop = mask[y:y + height, x:x + width]
    side = max(width, height) + 8
    square = np.zeros((side, side), dtype=np.uint8)
    x0 = (side - width) // 2
    y0 = (side - height) // 2
    square[y0:y0 + height, x0:x0 + width] = crop
    silhouette = cv2.resize(square, (16, 16), interpolation=cv2.INTER_AREA)
    silhouette = (silhouette.astype(np.float32) / 255.0).reshape(-1) * 0.35

    # Canonicalize the observed contour from camera pixels only: align its
    # principal axis horizontally and put the heavier end on the left. This
    # exposes the rounded spoon head versus the rectangular fork head while
    # remaining invariant to the object's yaw on either table.
    coordinates = points.reshape(-1, 2).astype(np.float32)
    centre_xy = coordinates.mean(axis=0)
    covariance = np.cov((coordinates - centre_xy).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    principal_angle = math.degrees(math.atan2(
        float(principal[1]), float(principal[0])))
    rotation = cv2.getRotationMatrix2D(
        (float(centre_xy[0]), float(centre_xy[1])), principal_angle, 1.0)
    aligned = cv2.warpAffine(
        mask, rotation, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST)
    aligned_points = cv2.findNonZero(aligned)
    ax, ay, aw, ah = cv2.boundingRect(aligned_points)
    aligned_crop = aligned[ay:ay + ah, ax:ax + aw]
    midpoint = max(1, aligned_crop.shape[1] // 2)
    if (cv2.countNonZero(aligned_crop[:, midpoint:])
            > cv2.countNonZero(aligned_crop[:, :midpoint])):
        aligned_crop = cv2.flip(aligned_crop, 1)
    canonical = cv2.resize(
        aligned_crop, (32, 16), interpolation=cv2.INTER_AREA)
    canonical_weight = 0.90 * max(
        0.0, min(1.0, (raw_aspect - 1.5) / 1.5))
    canonical = (
        canonical.astype(np.float32) / 255.0).reshape(-1) * canonical_weight

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = max(cv2.contourArea(contour), 1.0)
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    aspect = raw_aspect
    extent = area / max(1.0, width * height)
    hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    solidity = area / hull_area
    hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    hu = np.clip(hu, -10.0, 10.0) / 10.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)
    geometry_colour = np.asarray([
        math.log(area) / 10.0,
        min(aspect, 8.0) / 8.0,
        circularity,
        extent,
        solidity,
        mean_h / 180.0,
        mean_s / 255.0,
        mean_v / 255.0,
    ], dtype=np.float32) * 3.0
    return np.concatenate((
        silhouette,
        canonical,
        geometry_colour,
        hu.astype(np.float32) * 0.25,
    ))


class SyntheticShapeKNN:
    """Supervised k-NN trained at startup on augmented synthetic silhouettes."""

    COLOURS = {
        'glass': (240, 184, 77),
        'wine_glass': (230, 171, 115),
        'plate': (240, 247, 250),
        'cup': (219, 184, 61),
        'spoon': (230, 222, 214),
        'fork': (227, 219, 212),
        'knife': (145, 145, 140),
        'bottle': (55, 150, 45),
        'bowl': (45, 105, 235),
        'napkin': (185, 85, 175),
    }

    def __init__(self, metres_per_pixel: float, samples_per_class: int = 32) -> None:
        self.metres_per_pixel = metres_per_pixel
        self.samples: Dict[str, np.ndarray] = {}
        rng = np.random.default_rng(7319)
        for name in CLASS_NAMES:
            rows = []
            for sample_index in range(samples_per_class):
                scale = float(rng.uniform(0.92, 1.08))
                angle = float(
                    sample_index * 360.0 / samples_per_class
                    + rng.uniform(-2.5, 2.5))
                brightness = float(rng.uniform(0.88, 1.08))
                mask, image = self._training_image(name, scale, angle, brightness)
                rows.append(_mask_feature(mask, image))
            self.samples[name] = np.vstack(rows)

    def _px(self, metres: float) -> int:
        return max(2, int(round(metres / self.metres_per_pixel)))

    def _training_image(
        self, name: str, scale: float, angle: float, brightness: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        size = 160
        centre = size // 2
        mask = np.zeros((size, size), dtype=np.uint8)
        colour = tuple(int(min(255, value * brightness)) for value in self.COLOURS[name])

        if name == 'plate':
            cv2.circle(mask, (centre, centre), self._px(0.11), 255, -1)
        elif name == 'glass':
            cv2.circle(mask, (centre, centre), self._px(0.0375), 255, -1)
        elif name == 'wine_glass':
            cv2.circle(mask, (centre, centre), self._px(0.049), 255, -1)
        elif name == 'cup':
            radius = self._px(0.055)
            cv2.circle(mask, (centre, centre), radius, 255, -1)
            cv2.rectangle(
                mask,
                (centre - self._px(0.018), centre + radius - 2),
                (centre + self._px(0.018), centre + radius + self._px(0.035)),
                255, -1,
            )
        elif name == 'spoon':
            half_width = self._px(0.009)
            cv2.rectangle(mask,
                          (centre - self._px(0.040), centre - half_width),
                          (centre + self._px(0.090), centre + half_width),
                          255, -1)
            cv2.ellipse(mask, (centre - self._px(0.065), centre),
                        (self._px(0.035), self._px(0.024)), 0, 0, 360, 255, -1)
        elif name == 'fork':
            half_width = self._px(0.009)
            cv2.rectangle(mask, (centre - self._px(0.035), centre - half_width),
                          (centre + self._px(0.095), centre + half_width), 255, -1)
            cv2.rectangle(mask, (centre - self._px(0.095), centre - self._px(0.022)),
                          (centre - self._px(0.035), centre + self._px(0.022)), 255, -1)
        elif name == 'knife':
            cv2.rectangle(mask, (centre - self._px(0.10), centre - self._px(0.014)),
                          (centre + self._px(0.01), centre + self._px(0.014)), 255, -1)
            cv2.rectangle(mask, (centre + self._px(0.01), centre - self._px(0.018)),
                          (centre + self._px(0.10), centre + self._px(0.018)), 255, -1)
        elif name == 'bottle':
            cv2.circle(mask, (centre, centre), self._px(0.040), 255, -1)
        elif name == 'bowl':
            cv2.circle(mask, (centre, centre), self._px(0.075), 255, -1)
        elif name == 'napkin':
            half = self._px(0.060)
            cv2.rectangle(mask, (centre - half, centre - half),
                          (centre + half, centre + half), 255, -1)
        else:
            raise KeyError(name)

        matrix = cv2.getRotationMatrix2D((centre, centre), angle, scale)
        mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[mask > 0] = colour
        return mask, image

    def class_costs(self, feature: np.ndarray, neighbours: int = 3) -> Dict[str, float]:
        result = {}
        for name, samples in self.samples.items():
            distances = np.sum((samples - feature) ** 2, axis=1)
            nearest = np.partition(distances, min(neighbours, len(distances)) - 1)[
                :neighbours]
            result[name] = float(np.mean(nearest))
        return result

    @staticmethod
    def calibrated_confidence(costs: Dict[str, float], assigned_name: str) -> float:
        """Convert relative class distances into a normalized confidence."""
        minimum = min(costs.values())
        temperature = 0.45
        weights = {
            name: math.exp(-min(60.0, (cost - minimum) / temperature))
            for name, cost in costs.items()
        }
        probability = weights[assigned_name] / sum(weights.values())
        return float(max(0.01, min(0.995, probability)))

    def assign_unique(self, features: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        return self.assign_subset(features, CLASS_NAMES)

    def assign_subset(
        self,
        features: Sequence[np.ndarray],
        allowed_names: Sequence[str],
    ) -> List[Tuple[str, float]]:
        """Assign a known subset uniquely, as needed on the destination table."""
        allowed_names = tuple(allowed_names)
        if len(features) != len(allowed_names):
            raise ValueError(
                f'expected {len(allowed_names)} observations, got {len(features)}')
        if len(set(allowed_names)) != len(allowed_names):
            raise ValueError('allowed_names contains duplicates')
        if not set(allowed_names) <= set(CLASS_NAMES):
            raise ValueError('allowed_names contains an unknown class')
        costs = [self.class_costs(feature) for feature in features]
        # Dynamic-programming assignment is O(N * 2^N), avoiding 10! brute
        # force now that three irrelevant visual classes are also present.
        states: Dict[int, Tuple[float, Tuple[str, ...]]] = {0: (0.0, ())}
        for observation_index in range(len(features)):
            next_states: Dict[int, Tuple[float, Tuple[str, ...]]] = {}
            for mask, (total, labels) in states.items():
                for class_index, name in enumerate(allowed_names):
                    bit = 1 << class_index
                    if mask & bit:
                        continue
                    new_mask = mask | bit
                    candidate = total + costs[observation_index][name]
                    current = next_states.get(new_mask)
                    if current is None or candidate < current[0]:
                        next_states[new_mask] = (candidate, labels + (name,))
            states = next_states
        full_mask = (1 << len(allowed_names)) - 1
        best_labels = states[full_mask][1]
        return [
            (name, self.calibrated_confidence(costs[index], name))
            for index, name in enumerate(best_labels)
        ]


def roi_pixels(projector: PixelProjector, bounds: dict) -> Tuple[int, int, int, int]:
    corners = [
        projector.world_to_pixel(x, y)
        for x in (float(bounds['x_min']), float(bounds['x_max']))
        for y in (float(bounds['y_min']), float(bounds['y_max']))
    ]
    u_values = [point[0] for point in corners]
    v_values = [point[1] for point in corners]
    x0 = max(0, int(math.floor(min(u_values))))
    y0 = max(0, int(math.floor(min(v_values))))
    x1 = min(projector.width, int(math.ceil(max(u_values))))
    y1 = min(projector.height, int(math.ceil(max(v_values))))
    return x0, y0, x1, y1


def segment_observations(
    image: np.ndarray,
    projector: PixelProjector,
    bounds: dict,
    background_distance: float,
    min_area: float,
    expected_count: int,
) -> Tuple[List[VisualObservation], np.ndarray, Tuple[int, int, int, int]]:
    """Segment objects from the source table without using ground-truth poses."""
    x0, y0, x1, y1 = roi_pixels(projector, bounds)
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return [], np.zeros(image.shape[:2], dtype=np.uint8), (x0, y0, x1, y1)
    background = np.median(roi.reshape(-1, 3), axis=0)
    difference = np.linalg.norm(roi.astype(np.float32) - background, axis=2)
    local_mask = (difference >= float(background_distance)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel)
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:expected_count]

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    observations: List[VisualObservation] = []
    for contour in contours:
        shifted = contour + np.asarray([[[x0, y0]]], dtype=contour.dtype)
        cv2.drawContours(full_mask, [shifted], -1, 255, -1)
        # The centre of the oriented bounding rectangle is a better estimate
        # of the SDF model origin than the centre of mass for spoon / fork.
        centroid = tuple(map(float, cv2.minAreaRect(shifted)[0]))
        bx, by, bw, bh = cv2.boundingRect(shifted)
        component_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(component_mask, [shifted], -1, 255, -1)
        observations.append(VisualObservation(
            contour=shifted,
            centroid=centroid,
            bbox=(bx, by, bw, bh),
            feature=_mask_feature(component_mask, image),
        ))
    return observations, full_mask, (x0, y0, x1, y1)
