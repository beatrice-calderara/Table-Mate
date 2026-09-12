# Table Mate

Table Mate is a simulated robotic system for visual object recognition and autonomous table setting. It combines **ROS 2**, **Gazebo Harmonic**, **OpenCV computer vision**, supervised **k-nearest neighbors classification**, analytical inverse kinematics, and joint-space control.

A five-axis robotic arm equipped with a parallel gripper recognizes tableware through overhead RGB cameras, picks selected objects from a source table, arranges them on a dining table, and can later return every object to its initial position.

> Project status: version `1.3.0` — ROS 2 package `cv_arm_table_setting_demo`.

## Table of contents

- [Features](#features)
- [Available commands](#available-commands)
- [System architecture](#system-architecture)
- [Computer vision pipeline](#computer-vision-pipeline)
- [Kinematics, motion generation, and control](#kinematics-motion-generation-and-control)
- [Simulation environment](#simulation-environment)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the demo](#running-the-demo)
- [ROS 2 interfaces](#ros-2-interfaces)
- [Configuration](#configuration)
- [Testing](#testing)
- [Repository structure](#repository-structure)


## Features

- Complete physics simulation in Gazebo Harmonic.
- Articulated robot arm with five actuated joints:
  - base yaw;
  - shoulder pitch;
  - elbow pitch;
  - wrist pitch;
  - gripper roll.
- Parallel gripper with two prismatic fingers.
- Two `640 × 480` overhead RGB cameras:
  - a source camera used to locate objects before picking;
  - a destination camera used by the `clean table` operation.
- Foreground segmentation based on color distance from the table background.
- Supervised k-NN classifier trained at startup using augmented synthetic shapes.
- Recognition of 10 object classes:
  - `glass`;
  - `wine_glass`;
  - `plate`;
  - `cup`;
  - `spoon`;
  - `fork`;
  - `knife`;
  - `bottle`;
  - `bowl`;
  - `napkin`.
- Globally unique class assignment through dynamic programming.
- Pixel-to-world conversion using a pinhole camera model.
- Multi-frame detection stability checks.
- Analytical inverse kinematics with joint-limit validation.
- Smooth cosine-interpolated joint trajectories at 50 Hz.
- Explicit utensil-orientation handling during transfer.
- Reliable grasp and placement using Gazebo `DetachableJoint` plugins.
- Three predefined meals and arbitrary custom object selections.
- Automatic scene restoration.
- 57 automated tests covering configuration, kinematics, vision, and the SDF model.

## Available commands

### Predefined meals

| Command | Objects |
|---|---|
| `breakfast` | `plate`, `cup`, `spoon` |
| `lunch` | `plate`, `fork`, `glass` |
| `dinner` | `plate`, `fork`, `knife`, `wine_glass` |

Predefined meals use the traditional place settings stored in `config/scene.yaml`: the plate is placed in the center, the fork on the diner's left, the knife and spoon on the right, and the cup or glass in the upper-right area.

### Custom selections

The controller also accepts a comma-separated list of English object names:

```text
bottle, bowl, napkin
```

Custom mode provides a separate destination slot for every class. All ten objects can therefore be selected at the same time without nominal slot overlap.

Commands are case-insensitive, surrounding whitespace is removed, and `wine glass` is normalized to `wine_glass`. Duplicate or unknown names are rejected and reported on the status topic.

### Clearing the table

The command:

```text
clean table
```

enables the destination camera, recognizes the objects placed on the dining table, and returns them to their original source positions in reverse transfer order. The shorter command `clean` is accepted as an alias.

The controller runs only one task at a time. If the destination table is occupied, a new meal is rejected until `clean table` has completed.

## System architecture

```mermaid
flowchart LR
    GZ[Gazebo Harmonic<br/>world, robot, and sensors]
    C1[Source RGB camera]
    C2[Destination RGB camera]
    B[ros_gz_bridge<br/>ros_gz_image]
    P1[Source perception node]
    P2[Destination perception node]
    CMD[/table_setting/command]
    CTRL[Arm controller node]
    ACT[Joint position<br/>controllers]
    DJ[DetachableJoint<br/>grasp and table locks]

    GZ --> C1 --> B --> P1
    GZ --> C2 --> B --> P2
    P1 -->|Detection3DArray| CTRL
    P2 -->|Detection3DArray| CTRL
    CMD --> CTRL
    CTRL -->|joint targets| B --> ACT --> GZ
    CTRL -->|attach / detach| B --> DJ --> GZ
```

The launch file starts:

1. the Gazebo server with OGRE rendering;
2. the Gazebo GUI after a four-second delay, unless headless mode is enabled;
3. the bridge for joint commands and `attach`/`detach` events;
4. the two RGB image bridges;
5. two perception-node instances, one for each camera;
6. the arm controller;
7. two `rqt_image_view` windows for annotated images, when enabled.

The world initially starts paused. After three seconds, the launch file sends the unpause request through Gazebo Transport. This gives the controller enough time to initialize the arm, open the gripper, and configure the object constraints.

## Computer vision pipeline

The perception pipeline is implemented in `vision_core.py` and `perception_node.py`. It does not use the nominal YAML positions to localize the objects. Configured poses are used only by the controller when restoring the scene and to retrieve known object dimensions and grasp heights.

### 1. Region of interest

Each camera has a region of interest defined on the table plane. `PixelProjector` uses the camera position, height, resolution, and horizontal field of view to convert the world-space ROI boundaries into image coordinates.

### 2. Foreground segmentation

The median color inside the ROI is treated as the background. A pixel belongs to the foreground when its Euclidean BGR distance from that background exceeds `background_distance`. Morphological cleanup is then applied, and contours smaller than `min_component_area` are discarded.

### 3. Visual descriptor

Each connected component is represented by a descriptor containing:

- a normalized `16 × 16` silhouette;
- a canonical `32 × 16` silhouette aligned to its principal axis;
- area and aspect ratio;
- circularity, extent, and solidity;
- mean HSV color;
- seven Hu moments.

PCA-based alignment and normalization of the heavier end of the silhouette help distinguish visually similar objects—especially the spoon, fork, and knife—while keeping the descriptor robust to yaw rotations on either table.

### 4. Synthetic training and classification

At startup, `SyntheticShapeKNN` generates 32 samples per class with randomized scale, rotation, and brightness. The mean distance from the three nearest neighbors produces the classification cost for each class.

Because the number and identity of the expected objects are known, labels are assigned globally and uniquely. Dynamic programming solves the assignment problem in `O(N · 2^N)` time, avoiding a brute-force search over `N!` permutations.

### 5. Stability and output

Detections are published only when:

- the number of components matches the expected count;
- the same class set remains visible;
- each object moves by less than 2 cm between consecutive frames;
- the condition remains valid for the configured number of frames—three by default.

Results are published both as `vision_msgs/msg/Detection3DArray` and as compact JSON. Annotated images display green bounding boxes and object labels.

## Kinematics, motion generation, and control

### Kinematic model

The robot consists of:

- a base rotating about the world `Z` axis;
- a planar shoulder-elbow-wrist chain;
- a `0.68 m` upper arm;
- a `0.62 m` forearm;
- a `0.16 m` wrist-to-gripper offset.

Inverse kinematics is solved analytically in `kinematics.py`. For a Cartesian target `(x, y, z)`:

1. base yaw is computed with `atan2`;
2. the wrist center is obtained by removing the gripper offset;
3. the shoulder and elbow angles are found using the law of cosines;
4. wrist pitch keeps the gripper vertical;
5. the solution is checked against every joint limit.

If the desired transfer plane cannot be reached at both endpoints, `find_reachable_safe_height` lowers it in one-centimeter steps until it finds the highest shared safe height above the required minimum.

### Motion generation

Joint targets are interpolated at 50 Hz with cosine blending:

```text
s(t) = 0.5 - 0.5 cos(πt)
```

The profile has zero velocity at both ends of the movement. Duration depends on the largest joint displacement and the `speed_scale` parameter, which is clamped internally to `[0.25, 1.5]`.

### Pick-and-place sequence

For each object, the controller:

1. reads the XY position measured by the RGB camera;
2. computes a reachable transfer height;
3. moves above the pick position;
4. opens the gripper and descends to the grasp height;
5. closes the fingers until contact;
6. attaches the object to the gripper using `DetachableJoint`;
7. releases the object's source-table constraint;
8. applies a small controlled grip compression;
9. lifts and transfers the object;
10. descends into its destination slot;
11. opens the gripper, detaches the object, and locks it to the table;
12. returns to the safe height.

The bottle uses a dedicated sequence: its gripper constraint is enabled before finger contact so that a small camera-centering error cannot knock it over. Utensils use a directed head-to-handle yaw reference, preventing an otherwise equivalent 180-degree rotation from reversing their placement.

### Gazebo control

Each joint is driven by a Gazebo `JointPositionController` with dedicated PD gains. ROS 2 publishes scalar `std_msgs/msg/Float64` targets, which the parameter bridge converts into Gazebo messages.

Three types of `DetachableJoint` constraint are available for each object:

- gripper-to-object;
- source-table-to-object;
- destination-table-to-object.

These constraints make grasping, transfer, and placement deterministic while keeping all objects dynamic in the simulation.

## Simulation environment

`worlds/table_setting.sdf` contains:

- a floor;
- the source table;
- the destination table and placemat;
- a dining chair;
- two overhead RGB cameras;
- the articulated robot;
- ten dynamic table objects;
- Gazebo systems for physics, sensors, scene broadcasting, and user commands;
- position controllers for the five arm joints and two gripper fingers;
- grasp and table-lock plugins for every selectable object.

The Gazebo GUI configuration is embedded directly in the SDF world.

## Requirements

Reference environment:

- Ubuntu 24.04;
- ROS 2 Jazzy;
- Gazebo Harmonic;
- Python 3;
- OpenCV and NumPy;
- X11/XWayland for the Gazebo and `rqt_image_view` windows.

ROS dependencies declared by the package:

- `ament_index_python`;
- `launch` and `launch_ros`;
- `rclpy`;
- `ros_gz_sim`;
- `ros_gz_bridge`;
- `ros_gz_image`;
- `rqt_image_view`;
- `cv_bridge`;
- `sensor_msgs`;
- `std_msgs`;
- `vision_msgs`.

Python dependencies:

- `PyYAML`;
- `NumPy`;
- OpenCV;
- `pytest` for testing only.

### Parallels and Apple Silicon note

The current launch file was designed for a Parallels VM running on Apple Silicon. It expects a locally built OGRE 1 backend at:

```text
~/gazebo_ogre1/install
```

It explicitly sets `GZ_RENDERING_ENGINE_PATH`, `LD_LIBRARY_PATH`, `OGRE_PLUGIN_DIR`, `QT_QPA_PLATFORM=xcb`, and `QT_X11_NO_MITSHM=1`. This repository contains OGRE shaders, fonts, and rendering resources, but it does **not** contain the rendering-engine binaries.

Before running the project, either:

- install the backend at the expected path;
- change `ogre_prefix` in `launch/demo.launch.py` to match your installation; or
- adapt the launch file to use the standard rendering backend available on your platform.

## Installation

### 1. Set up ROS 2

Install ROS 2 Jazzy by following the official ROS documentation, then source it:

```bash
source /opt/ros/jazzy/setup.bash
```

### 2. Create a workspace and clone the repository

```bash
mkdir -p ~/table_mate_ws/src
cd ~/table_mate_ws/src
git clone https://github.com/beatrice-calderara/Table-Mate.git
```

### 3. Install dependencies

From a terminal where ROS 2 has already been sourced:

```bash
cd ~/table_mate_ws
sudo rosdep init  # Run only if rosdep has never been initialized
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

If required, the main packages can be installed explicitly:

```bash
sudo apt update
sudo apt install \
  python3-colcon-common-extensions \
  python3-numpy \
  python3-opencv \
  python3-pytest \
  python3-yaml \
  ros-jazzy-cv-bridge \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-image \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-rqt-image-view \
  ros-jazzy-vision-msgs
```

### 4. Build the workspace

```bash
cd ~/table_mate_ws
colcon build --symlink-install
source install/setup.bash
```

Run `source install/setup.bash` in every new terminal after sourcing ROS 2.

## Running the demo

### Interactive mode

Start Gazebo, both perception pipelines, the controller, and the image-debug windows:

```bash
ros2 launch cv_arm_table_setting_demo demo.launch.py
```

Wait for the following message:

```text
READY - send breakfast, lunch, dinner, an English list like "bottle, cup", or clean table
```

Open a second terminal, source the environment, and publish a command:

```bash
source /opt/ros/jazzy/setup.bash
source ~/table_mate_ws/install/setup.bash

ros2 topic pub --once /table_setting/command \
  std_msgs/msg/String "{data: 'breakfast'}"
```

Additional examples:

```bash
# Lunch
ros2 topic pub --once /table_setting/command \
  std_msgs/msg/String "{data: 'lunch'}"

# Dinner
ros2 topic pub --once /table_setting/command \
  std_msgs/msg/String "{data: 'dinner'}"

# Custom selection
ros2 topic pub --once /table_setting/command \
  std_msgs/msg/String "{data: 'bottle, bowl, napkin'}"

# Restore the scene
ros2 topic pub --once /table_setting/command \
  std_msgs/msg/String "{data: 'clean table'}"
```

### Run a command at startup

The `meal` launch argument accepts a predefined meal, `clean table`, or a custom list:

```bash
ros2 launch cv_arm_table_setting_demo demo.launch.py meal:=dinner
```

Quote lists containing spaces or commas:

```bash
ros2 launch cv_arm_table_setting_demo demo.launch.py \
  meal:="bottle, cup, napkin"
```

### Launch arguments

| Argument | Default | Description |
|---|---:|---|
| `meal` | `none` | Initial meal, object list, or `clean table` command |
| `headless` | `false` | When `true`, does not open the Gazebo GUI; intended for automated tests |
| `speed_scale` | `1.0` | Motion speed scale, internally limited to `[0.25, 1.5]` |
| `show_camera` | `true` | Opens both annotated RGB streams in `rqt_image_view` |

Headless example:

```bash
ros2 launch cv_arm_table_setting_demo demo.launch.py \
  headless:=true show_camera:=false meal:=lunch
```

### Monitor the controller

```bash
ros2 topic echo /table_setting/status
```

Typical states include `STARTING`, `READY`, `RUNNING`, `MOVING_TO_PICK`, `ATTACHING`, `LIFTING`, `MOVING_TO_PLACE`, `COMPLETED`, `CLEANING`, `CLEANED`, `BUSY`, and `ERROR`.

## ROS 2 interfaces

### User commands and status

| Topic | Type | Controller direction | Purpose |
|---|---|---|---|
| `/table_setting/command` | `std_msgs/msg/String` | Input | Meal, custom list, or clean command |
| `/table_setting/status` | `std_msgs/msg/String` | Output | Global robot-cycle state |

### Images and perception

| Topic | Type | Description |
|---|---|---|
| `/table_setting/camera/image` | `sensor_msgs/msg/Image` | Source-table RGB image |
| `/table_setting/destination_camera/image` | `sensor_msgs/msg/Image` | Destination-table RGB image |
| `/table_setting/vision/debug_image` | `sensor_msgs/msg/Image` | Annotated source image |
| `/table_setting/vision/destination_debug_image` | `sensor_msgs/msg/Image` | Annotated destination image |
| `/table_setting/detections` | `vision_msgs/msg/Detection3DArray` | Source-table classes and poses |
| `/table_setting/destination_detections` | `vision_msgs/msg/Detection3DArray` | Destination-table classes and poses |
| `/table_setting/recognized_objects` | `std_msgs/msg/String` | Source detections as JSON |
| `/table_setting/destination_recognized_objects` | `std_msgs/msg/String` | Destination detections as JSON |
| `/table_setting/vision/status` | `std_msgs/msg/String` | Source-perception status |
| `/table_setting/vision/destination_status` | `std_msgs/msg/String` | Destination-perception status |

Each JSON entry contains an ID, class, confidence, camera role, pixel bounding box, world position, and selection metadata.

### Joint commands

All joint topics use `std_msgs/msg/Float64`:

```text
/table_setting/joint/base_yaw
/table_setting/joint/shoulder_pitch
/table_setting/joint/elbow_pitch
/table_setting/joint/wrist_pitch
/table_setting/joint/gripper_roll
/table_setting/joint/left_finger
/table_setting/joint/right_finger
```

### Grasp and table constraints

For every recognized `<object>`, the simulation exposes `std_msgs/msg/Empty` topics following this pattern:

```text
/table_setting/grip/<object>/attach
/table_setting/grip/<object>/detach
/table_setting/source_hold/<object>/attach
/table_setting/source_hold/<object>/detach
/table_setting/destination_hold/<object>/attach
/table_setting/destination_hold/<object>/detach
```

These topics are internal simulation interfaces and are managed automatically by the controller.

## Configuration

Most system behavior is centralized in `config/scene.yaml`.

### `vision`

This section defines:

- image and debug topics;
- camera position, resolution, and horizontal field of view;
- source and destination ROIs;
- table-plane height;
- segmentation threshold;
- minimum component area;
- expected object counts;
- required stable-frame count;
- the three additional selectable objects.

### `robot`

This section defines:

- base pose;
- link lengths;
- safe transfer height;
- grasp and release clearances;
- gripper opening;
- home configuration;
- joint limits.

### Objects

Each object specifies:

- class label;
- initial pose;
- dimensions;
- grasp width;
- upper-surface height used for picking.

Objects used by predefined meals are stored in `objects`. The `bottle`, `bowl`, and `napkin` entries are stored under `vision.distractors`, but they are still recognized and can be selected in custom commands.

### Destination slots and recipes

- `destination_slots`: traditional positions used by predefined meals;
- `custom_destination_slots`: separate positions used by custom selections;
- `recipes`: the object lists for breakfast, lunch, and dinner.

At startup, `validate_scene` checks object counts, required fields, recipes, destination slots, grasp heights, and the home configuration.

> If you change object poses or dimensions in the YAML file, update the corresponding models and poses in `worlds/table_setting.sdf` as well.

## Testing

The repository contains 57 tests in four areas:

- `test_kinematics.py`: inverse kinematics, reachability, limits, grasping, and orientation;
- `test_layout.py`: command parsing, recipes, slots, and YAML validation;
- `test_sdf_names.py`: world integrity, models, joints, plugins, and configuration consistency;
- `test_vision.py`: projection, synthetic classification, SDF silhouettes, and the complete RGB pipeline.

After building the workspace:

```bash
cd ~/table_mate_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

colcon test --packages-select cv_arm_table_setting_demo
colcon test-result --verbose
```

To run the Python tests directly from the repository root:

```bash
python3 -m pytest -q
```

If `pytest` belongs to a different Python environment and cannot find the local package:

```bash
PYTHONPATH=. pytest -q
```

Verified result for the current repository revision:

```text
57 passed
```

## Repository structure

```text
Table-Mate/
├── config/
│   └── scene.yaml                 # Scene, objects, robot, slots, and meals
├── cv_arm_table_setting_demo/
│   ├── controller_node.py         # Pick/place and clean-table state machine
│   ├── kinematics.py              # IK and gripper orientation helpers
│   ├── layout.py                  # Scene loading and validation
│   ├── perception_node.py         # ROS 2 RGB-perception node
│   └── vision_core.py             # Segmentation, descriptors, and k-NN
├── launch/
│   └── demo.launch.py             # Gazebo, bridges, and node orchestration
├── ogre/                          # Rendering resources, fonts, and shaders
├── resource/
│   └── cv_arm_table_setting_demo  # Ament-index package marker
├── test/
│   ├── test_kinematics.py
│   ├── test_layout.py
│   ├── test_sdf_names.py
│   └── test_vision.py
├── worlds/
│   └── table_setting.sdf          # Gazebo world, robot, objects, and plugins
├── package.xml                    # ROS 2 metadata and dependencies
├── setup.cfg                      # ROS 2 entry-point installation
└── setup.py                       # ament_python package definition
```

Installed console entry points:

```text
perception -> cv_arm_table_setting_demo.perception_node:main
controller -> cv_arm_table_setting_demo.controller_node:main
```

