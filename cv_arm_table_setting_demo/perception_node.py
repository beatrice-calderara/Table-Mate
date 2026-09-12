"""RGB computer vision and supervised ML recognition for the Gazebo scene."""

from __future__ import annotations

import json
from typing import Dict, Tuple

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .layout import load_scene, scene_objects
from .vision_core import (
    BOUNDING_BOX_COLOUR,
    ADDITIONAL_CLASS_NAMES,
    CLASS_NAMES,
    DISPLAY_NAMES,
    PixelProjector,
    SyntheticShapeKNN,
    segment_observations,
)


class VisualMLPerception(Node):
    """Recognize objects from RGB pixels and publish their estimated world poses."""

    def __init__(self) -> None:
        super().__init__('visual_ml_perception')
        default_config = (
            get_package_share_directory('cv_arm_table_setting_demo')
            + '/config/scene.yaml'
        )
        self.declare_parameter('scene_config', default_config)
        self.declare_parameter('camera_role', 'source')
        self.scene = load_scene(self.get_parameter('scene_config').value)
        self.objects = scene_objects(self.scene)
        vision = self.scene['vision']
        self.role = str(self.get_parameter('camera_role').value).strip().lower()
        if self.role not in ('source', 'destination'):
            raise ValueError('camera_role must be source or destination')
        self.is_destination = self.role == 'destination'
        camera = vision['destination_camera'] if self.is_destination else vision['camera']
        self.bounds = (
            vision['destination_roi'] if self.is_destination
            else vision['source_roi']
        )
        image_topic = (
            vision['destination_image_topic'] if self.is_destination
            else vision['image_topic']
        )
        debug_topic = (
            vision['destination_debug_topic'] if self.is_destination
            else vision['debug_topic']
        )
        self.projector = PixelProjector(
            camera_x=float(camera['x']),
            camera_y=float(camera['y']),
            camera_z=float(camera['z']),
            plane_z=float(vision['table_plane_z']),
            width=int(camera['width']),
            height=int(camera['height']),
            horizontal_fov=float(camera['horizontal_fov']),
        )
        self.classifier = SyntheticShapeKNN(self.projector.metres_per_pixel)
        self.bridge = CvBridge()
        self.background_distance = float(vision['background_distance'])
        self.min_area = float(vision['min_component_area'])
        self.expected_names = [] if self.is_destination else list(CLASS_NAMES)
        self.expected_count = len(self.expected_names)
        self.required_stable_frames = int(vision['stable_frames'])
        self.previous: Dict[str, Tuple[float, float]] = {}
        self.stable_frames = 0
        self.capture_enabled = not self.is_destination

        self.detections_pub = self.create_publisher(
            Detection3DArray,
            ('/table_setting/destination_detections' if self.is_destination
             else '/table_setting/detections'), 10)
        self.json_pub = self.create_publisher(
            String,
            ('/table_setting/destination_recognized_objects' if self.is_destination
             else '/table_setting/recognized_objects'), 10)
        self.status_pub = self.create_publisher(
            String,
            ('/table_setting/vision/destination_status' if self.is_destination
             else '/table_setting/vision/status'), 10)
        self.debug_pub = self.create_publisher(
            Image, str(debug_topic), qos_profile_sensor_data)
        self.create_subscription(
            Image, str(image_topic), self.image_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            String, '/table_setting/status', self.controller_status_callback, 10)
        self.get_logger().info(
            f'AI vision {self.role} ready: RGB camera + segmentation + k-NN')

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.capture_enabled:
                cv2.putText(
                    image, f'AI CV {self.role} paused', (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
                self.publish_debug(image, msg)
                return
            observations, _mask, _roi = segment_observations(
                image=image,
                projector=self.projector,
                bounds=self.bounds,
                background_distance=self.background_distance,
                min_area=self.min_area,
                expected_count=self.expected_count,
            )
            annotated = image.copy()
            if len(observations) != self.expected_count:
                self.stable_frames = 0
                cv2.putText(
                    annotated,
                    f'CV: {len(observations)}/{self.expected_count} objects',
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                self.status_pub.publish(String(
                    data=f'WAITING_IMAGE_OBJECTS {len(observations)}/{self.expected_count}'))
                self.publish_debug(annotated, msg)
                return

            assignments = self.classifier.assign_subset(
                [observation.feature for observation in observations],
                self.expected_names,
            )
            recognized: Dict[str, Tuple[float, float, float, object]] = {}
            for observation, (name, confidence) in zip(observations, assignments):
                world_x, world_y = self.projector.pixel_to_world(*observation.centroid)
                recognized[name] = (world_x, world_y, confidence, observation)
                bx, by, bw, bh = observation.bbox
                cv2.rectangle(
                    annotated, (bx, by), (bx + bw, by + bh),
                    BOUNDING_BOX_COLOUR, 2)
                cv2.putText(
                    annotated,
                    DISPLAY_NAMES[name],
                    (bx, max(18, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    BOUNDING_BOX_COLOUR, 1)

            positions = {name: (values[0], values[1]) for name, values in recognized.items()}
            if self.is_stable(positions):
                self.stable_frames += 1
            else:
                self.stable_frames = 1
            self.previous = positions
            cv2.putText(
                annotated,
                f'AI CV {self.role}: {self.expected_count} objects - stability '
                f'{self.stable_frames}/{self.required_stable_frames}',
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (30, 180, 30), 2)

            if self.stable_frames >= self.required_stable_frames:
                self.publish_detections(msg, recognized)
                self.status_pub.publish(String(
                    data=(f'VISION_READY {self.role} '
                          f'{self.expected_count}/{self.expected_count}')))
            self.publish_debug(annotated, msg)
        except Exception as exc:  # keep the camera subscription alive after bad frames
            self.get_logger().error(f'Computer vision pipeline error: {exc}')
            self.status_pub.publish(String(data=f'VISION_ERROR {exc}'))

    def controller_status_callback(self, msg: String) -> None:
        if msg.data.startswith('RUNNING'):
            if self.is_destination and 'objects:' in msg.data:
                payload = msg.data.split('objects:', 1)[1]
                self.expected_names = [
                    name.strip() for name in payload.split(',') if name.strip()
                ]
                self.expected_count = len(self.expected_names)
            self.capture_enabled = False
            self.stable_frames = 0
        elif msg.data.startswith('COMPLETED'):
            self.capture_enabled = self.is_destination and bool(self.expected_names)
            self.previous = {}
            self.stable_frames = 0
        elif msg.data.startswith('CLEANING'):
            self.capture_enabled = False
            self.stable_frames = 0
        elif msg.data.startswith(('READY', 'CLEANED')):
            self.capture_enabled = not self.is_destination
            self.previous = {}
            self.stable_frames = 0

    def is_stable(self, positions: Dict[str, Tuple[float, float]]) -> bool:
        if positions.keys() != self.previous.keys():
            return False
        return all(
            (x - self.previous[name][0]) ** 2
            + (y - self.previous[name][1]) ** 2 < 0.02 ** 2
            for name, (x, y) in positions.items()
        )

    def publish_detections(self, image_msg: Image, recognized: dict) -> None:
        output = Detection3DArray()
        output.header = image_msg.header
        output.header.frame_id = self.scene['world_frame']
        compact = []
        for name, (world_x, world_y, confidence, observation) in recognized.items():
            additional = name in ADDITIONAL_CLASS_NAMES
            if additional:
                item = self.scene['vision']['distractors'][name]
                label = str(item['label'])
                size = item['size']
                z = float(item['pose']['z'])
            else:
                obj = self.objects[name]
                label = obj.label
                size = {'x': obj.size.x, 'y': obj.size.y, 'z': obj.size.z}
                z = obj.pose.z
            compact.append({
                'id': name,
                'class': label,
                'confidence': round(float(confidence), 4),
                'selectable': True,
                'fixed_meal_object': not additional,
                'source': 'rgb_camera_synthetic_knn',
                'camera_role': self.role,
                'pixel_bbox': list(observation.bbox),
                'position': {'x': world_x, 'y': world_y, 'z': z},
            })
            detection = Detection3D()
            detection.header = output.header
            detection.id = name
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = label
            result.hypothesis.score = float(confidence)
            result.pose.pose.position.x = world_x
            result.pose.pose.position.y = world_y
            result.pose.pose.position.z = z
            result.pose.pose.orientation.w = 1.0
            detection.results.append(result)
            detection.bbox.center.position.x = world_x
            detection.bbox.center.position.y = world_y
            detection.bbox.center.position.z = z
            detection.bbox.center.orientation.w = 1.0
            detection.bbox.size.x = float(size['x'])
            detection.bbox.size.y = float(size['y'])
            detection.bbox.size.z = float(size['z'])
            output.detections.append(detection)
        self.detections_pub.publish(output)
        self.json_pub.publish(String(data=json.dumps(compact, ensure_ascii=False)))

    def publish_debug(self, image, source_msg: Image) -> None:
        debug = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        debug.header = source_msg.header
        self.debug_pub.publish(debug)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualMLPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
