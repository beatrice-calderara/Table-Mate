from pathlib import Path
import math
import xml.etree.ElementTree as ET

from cv_arm_table_setting_demo.layout import load_scene, scene_objects


WORLD = Path(__file__).parents[1] / 'worlds' / 'table_setting.sdf'
CONFIG = Path(__file__).parents[1] / 'config' / 'scene.yaml'
LAUNCH = Path(__file__).parents[1] / 'launch' / 'demo.launch.py'


def test_model_children_have_unique_names():
    root = ET.parse(WORLD).getroot()
    for model in root.findall('.//model'):
        named_children = [
            child.attrib['name']
            for child in model
            if child.tag in {'link', 'joint', 'frame', 'model'} and 'name' in child.attrib
        ]
        assert len(named_children) == len(set(named_children)), (
            f'nomi SDF duplicati nel modello {model.attrib.get("name")}: {named_children}'
        )


def test_finger_joint_names_differ_from_finger_link_names():
    root = ET.parse(WORLD).getroot()
    arm = next(model for model in root.findall('.//model')
               if model.attrib.get('name') == 'table_setting_arm')
    links = {item.attrib['name'] for item in arm.findall('link')}
    joints = {item.attrib['name'] for item in arm.findall('joint')}
    assert links.isdisjoint(joints)


def test_dynamic_collisions_do_not_use_ellipsoid_trimeshes():
    root = ET.parse(WORLD).getroot()
    # DART's ODE backend on ARM64 may abort in dCollideBTL when an ellipsoid
    # collision proxy touches a table box. Ellipsoids remain allowed visually.
    assert not root.findall('.//collision/geometry/ellipsoid')


def test_world_loads_user_commands_for_deterministic_pose_correction():
    root = ET.parse(WORLD).getroot()
    plugins = {
        plugin.attrib.get('filename')
        for plugin in root.findall('./world/plugin')
    }
    assert 'gz-sim-user-commands-system' in plugins


def test_gripper_palm_covers_the_full_finger_travel():
    root = ET.parse(WORLD).getroot()
    arm = next(model for model in root.findall('.//model')
               if model.attrib.get('name') == 'table_setting_arm')
    palm = next(link for link in arm.findall('link')
                if link.attrib.get('name') == 'gripper_palm')
    palm_width = float(palm.find('visual/geometry/box/size').text.split()[1])
    finger_offset = 0.012
    maximum_travel = 0.11
    assert palm_width / 2.0 >= finger_offset + maximum_travel


def test_world_contains_every_configured_table_object():
    root = ET.parse(WORLD).getroot()
    world_models = {model.attrib['name'] for model in root.findall('./world/model')}
    assert {'glass', 'wine_glass', 'plate', 'cup', 'spoon', 'fork', 'knife'} <= world_models
    assert {'bottle', 'bowl', 'napkin'} <= world_models


def test_every_selectable_object_is_connected_to_a_gripper_plugin():
    root = ET.parse(WORLD).getroot()
    child_models = {
        plugin.find('child_model').text
        for plugin in root.findall('.//plugin')
        if plugin.find('child_model') is not None
    }
    assert set(scene_objects(load_scene(CONFIG))) <= child_models


def test_every_selectable_object_is_dynamic():
    root = ET.parse(WORLD).getroot()
    models = {model.attrib['name']: model for model in root.findall('./world/model')}
    for name in scene_objects(load_scene(CONFIG)):
        static = models[name].find('static')
        assert static is None or static.text.strip() != 'true'


def test_chair_is_on_the_side_opposite_the_robot():
    root = ET.parse(WORLD).getroot()
    models = {model.attrib['name']: model for model in root.findall('./world/model')}
    chair_x = float(models['dining_chair'].find('pose').text.split()[0])
    table_x = float(models['destination_table'].find('pose').text.split()[0])
    arm_x = float(models['table_setting_arm'].find('pose').text.split()[0])
    assert arm_x < table_x < chair_x
    assert models['dining_chair'].find('static').text.strip() == 'true'


def test_placemat_uses_the_selected_opaque_purple_red_colour():
    root = ET.parse(WORLD).getroot()
    table = next(model for model in root.findall('./world/model')
                 if model.attrib.get('name') == 'destination_table')
    placemat = next(visual for visual in table.findall('link/visual')
                    if visual.attrib.get('name') == 'placemat')
    rgba = tuple(map(float, placemat.find('material/diffuse').text.split()))
    assert rgba == (0.72, 0.06, 0.25, 1.0)


def test_placemat_collision_matches_visual_surface():
    root = ET.parse(WORLD).getroot()
    table = next(model for model in root.findall('./world/model')
                 if model.attrib.get('name') == 'destination_table')
    link = table.find('link')
    visual = next(item for item in link.findall('visual')
                  if item.attrib.get('name') == 'placemat')
    collision = next(item for item in link.findall('collision')
                     if item.attrib.get('name') == 'placemat')
    assert collision.find('pose').text == visual.find('pose').text
    assert (collision.find('geometry/box/size').text
            == visual.find('geometry/box/size').text)
    pose_z = float(collision.find('pose').text.split()[2])
    height = float(collision.find('geometry/box/size').text.split()[2])
    placemat_top = pose_z + height / 2.0
    scene = load_scene(CONFIG)
    assert all(abs(destination['surface_z'] - placemat_top) < 1e-9
               for destination in scene['destination_slots'].values())


def test_enlarged_placemat_covers_the_destination_camera_roi():
    root = ET.parse(WORLD).getroot()
    table = next(model for model in root.findall('./world/model')
                 if model.attrib.get('name') == 'destination_table')
    placemat = next(item for item in table.findall('link/visual')
                    if item.attrib.get('name') == 'placemat')
    size_x, size_y, _ = map(
        float, placemat.find('geometry/box/size').text.split())
    scene = load_scene(CONFIG)
    centre = scene['tables']['destination_center']
    roi = scene['vision']['destination_roi']
    assert abs((roi['x_max'] - roi['x_min']) - size_x) < 1e-9
    assert abs((roi['y_max'] - roi['y_min']) - size_y) < 1e-9
    assert abs((roi['x_min'] + roi['x_max']) / 2.0 - centre['x']) < 1e-9
    assert abs((roi['y_min'] + roi['y_max']) / 2.0 - centre['y']) < 1e-9


def test_closed_gripper_can_contact_thin_cutlery():
    root = ET.parse(WORLD).getroot()
    arm = next(model for model in root.findall('.//model')
               if model.attrib.get('name') == 'table_setting_arm')
    left = next(link for link in arm.findall('link')
                if link.attrib.get('name') == 'left_finger')
    offset = float(left.find('pose').text.split()[1])
    thickness = float(left.find('collision/geometry/box/size').text.split()[1])
    minimum_inner_gap = 2.0 * (offset - thickness / 2.0)
    assert abs(minimum_inner_gap) < 1e-9


def test_plate_uses_a_stable_visible_edge_grasp():
    root = ET.parse(WORLD).getroot()
    plate = next(model for model in root.findall('./world/model')
                 if model.attrib.get('name') == 'plate')
    collision_radius = float(
        plate.find('link/collision/geometry/cylinder/radius').text)
    visual_radius = float(
        plate.find('link/visual/geometry/cylinder/radius').text)
    scene = load_scene(CONFIG)
    assert scene['objects']['plate']['grasp_width'] == 2.0 * collision_radius
    assert collision_radius < visual_radius


def test_world_object_poses_match_perception_config():
    root = ET.parse(WORLD).getroot()
    models = {model.attrib['name']: model for model in root.findall('./world/model')}
    for obj in scene_objects(load_scene(CONFIG)).values():
        xyz = tuple(map(float, models[obj.name].find('pose').text.split()[:3]))
        assert xyz == (obj.pose.x, obj.pose.y, obj.pose.z)


def test_bottle_matches_glass_height_in_config_and_world():
    scene = load_scene(CONFIG)
    assert scene['vision']['distractors']['bottle']['size']['z'] == (
        scene['objects']['glass']['size']['z'])

    root = ET.parse(WORLD).getroot()
    models = {model.attrib['name']: model for model in root.findall('./world/model')}
    bottle_height = float(
        models['bottle'].find('link/collision/geometry/cylinder/length').text)
    glass_height = float(
        models['glass'].find('link/collision/geometry/cylinder/length').text)
    assert bottle_height == glass_height


def test_bottle_and_glasses_are_clear_of_the_bowl():
    objects = scene_objects(load_scene(CONFIG))
    bowl = objects['bowl'].pose
    for name in ('bottle', 'glass', 'wine_glass'):
        pose = objects[name].pose
        assert math.dist((pose.x, pose.y), (bowl.x, bowl.y)) > 0.25


def test_gripper_has_roll_joint_for_cutlery_orientation():
    root = ET.parse(WORLD).getroot()
    arm = next(model for model in root.findall('.//model')
               if model.attrib.get('name') == 'table_setting_arm')
    roll = next(joint for joint in arm.findall('joint')
                if joint.attrib.get('name') == 'gripper_roll')
    assert roll.find('axis/xyz').text.strip() == '1 0 0'


def test_world_has_real_rgb_camera_and_ogre_sensor_renderer():
    root = ET.parse(WORLD).getroot()
    world = root.find('world')
    sensors_plugin = next(
        plugin for plugin in world.findall('plugin')
        if plugin.attrib.get('filename') == 'gz-sim-sensors-system')
    assert sensors_plugin.find('render_engine').text.strip() == 'ogre'
    camera_model = next(model for model in world.findall('model')
                        if model.attrib.get('name') == 'overhead_camera')
    sensor = camera_model.find('link/sensor')
    assert sensor.attrib['type'] == 'camera'
    assert sensor.find('topic').text.strip() == '/table_setting/camera/image'
    assert sensor.find('camera/image/width').text.strip() == '640'
    assert sensor.find('camera/image/height').text.strip() == '480'
    destination_camera = next(
        model for model in world.findall('model')
        if model.attrib.get('name') == 'destination_overhead_camera')
    destination_sensor = destination_camera.find('link/sensor')
    assert destination_sensor.attrib['type'] == 'camera'
    assert destination_sensor.find('topic').text.strip() == (
        '/table_setting/destination_camera/image')
    scene = load_scene(CONFIG)
    assert scene['vision']['destination_image_topic'] == (
        '/table_setting/destination_camera/image')
    assert scene['vision']['destination_debug_topic'] == (
        '/table_setting/vision/destination_debug_image')


def test_server_and_gui_explicitly_force_ogre1():
    root = ET.parse(WORLD).getroot()
    assert root.find("world/plugin[@filename='gz-sim-sensors-system']/render_engine").text == 'ogre'
    assert root.find("world/gui/plugin[@filename='MinimalScene']/engine").text == 'ogre'
    launch_source = LAUNCH.read_text(encoding='utf-8')
    assert "--render-engine ogre" in launch_source
    assert "--render-engine ogre2" not in launch_source


def test_launch_forces_xcb_for_virtualized_qt_gui():
    launch_source = LAUNCH.read_text(encoding='utf-8')
    assert "SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb')" in launch_source
    assert "SetEnvironmentVariable('QT_X11_NO_MITSHM', '1')" in launch_source


def test_launch_starts_server_before_delayed_gui():
    launch_source = LAUNCH.read_text(encoding='utf-8')
    assert "'-s --render-engine ogre " in launch_source
    assert "cmd=['gz', 'sim', '-g', '-v', '3'" in launch_source
    assert 'condition=UnlessCondition(headless)' in launch_source


def test_camera_rate_is_safe_for_parallels_gui():
    root = ET.parse(WORLD).getroot()
    rates = [float(sensor.find('update_rate').text)
             for sensor in root.findall("world/model/link/sensor[@type='camera']")]
    assert rates == [5.0, 5.0]


def test_spoon_head_has_a_flat_non_rolling_collision_proxy():
    root = ET.parse(WORLD).getroot()
    spoon = next(model for model in root.findall('./world/model')
                 if model.attrib.get('name') == 'spoon')
    head = next(item for item in spoon.findall('link/collision')
                if item.attrib.get('name') == 'head')
    assert head.find('geometry/box') is not None
    assert head.find('geometry/sphere') is None


def test_tables_are_enlarged_but_placemat_size_is_unchanged():
    root = ET.parse(WORLD).getroot()
    models = {model.attrib['name']: model
              for model in root.findall('./world/model')}
    for name in ('source_table', 'destination_table'):
        top = next(item for item in models[name].findall('link/visual')
                   if item.attrib.get('name') == 'top')
        assert tuple(map(float, top.find('geometry/box/size').text.split())) == (
            1.10, 1.40, 0.08)
    placemat = next(
        item for item in models['destination_table'].findall('link/visual')
        if item.attrib.get('name') == 'placemat')
    assert tuple(map(
        float, placemat.find('geometry/box/size').text.split())) == (
            0.86, 1.00, 0.008)


def test_every_object_has_source_and_destination_table_holds():
    root = ET.parse(WORLD).getroot()
    scene = load_scene(CONFIG)
    expected = set(scene_objects(scene))
    models = {model.attrib['name']: model
              for model in root.findall('./world/model')}
    for table_name, prefix in (
        ('source_table', 'source_hold'),
        ('destination_table', 'destination_hold'),
    ):
        plugins = models[table_name].findall('plugin')
        assert {plugin.find('child_model').text for plugin in plugins} == expected
        for plugin in plugins:
            assert plugin.attrib['filename'] == (
                'gz-sim-detachable-joint-system')
            assert plugin.attrib['name'] == (
                'gz::sim::systems::DetachableJoint')
            child = plugin.find('child_model').text
            assert plugin.find('attach_topic').text == (
                f'/table_setting/{prefix}/{child}/attach')
            assert plugin.find('detach_topic').text == (
                f'/table_setting/{prefix}/{child}/detach')

    launch_source = LAUNCH.read_text(encoding='utf-8')
    assert '/table_setting/source_hold/{name}/attach' in launch_source
    assert '/table_setting/source_hold/{name}/detach' in launch_source
    assert '/table_setting/destination_hold/{name}/attach' in launch_source
    assert '/table_setting/destination_hold/{name}/detach' in launch_source


def test_fork_visual_is_one_connected_shape_for_segmentation():
    root = ET.parse(WORLD).getroot()
    fork = next(model for model in root.findall('./world/model')
                if model.attrib.get('name') == 'fork')
    visuals = {item.attrib['name']: item for item in fork.findall('link/visual')}
    handle = visuals['handle']
    head = visuals['head']
    handle_start = (
        float(handle.find('pose').text.split()[0])
        - float(handle.find('geometry/box/size').text.split()[0]) / 2.0)
    head_end = (
        float(head.find('pose').text.split()[0])
        + float(head.find('geometry/box/size').text.split()[0]) / 2.0)
    assert head_end >= handle_start
