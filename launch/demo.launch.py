import os
import sysconfig

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('cv_arm_table_setting_demo')
    ros_gz_dir = get_package_share_directory('ros_gz_sim')
    world = package_dir + '/worlds/table_setting.sdf'
    config = package_dir + '/config/scene.yaml'

    meal = LaunchConfiguration('meal')
    headless = LaunchConfiguration('headless')
    speed_scale = LaunchConfiguration('speed_scale')
    show_camera = LaunchConfiguration('show_camera')
    # Start the server first in every mode. On Parallels, launching server and
    # GUI together can race the first full-scene request and leave a responsive
    # 3-D window showing only the background. A delayed, separate GUI reliably
    # connects after SceneBroadcaster has published the world state.
    gz_args = PythonExpression([
        "'-s --render-engine ogre ", world, "'"
    ])

    # Parallels on Apple Silicon requires the locally built OGRE1 backend.
    # All shader resources are shipped by this package, so users don't need
    # to maintain rendering exports in their shell.
    ogre_prefix = os.path.expanduser('~/gazebo_ogre1/install')
    engine_path = os.path.join(
        ogre_prefix, 'lib', 'gz-rendering-8', 'engine-plugins')
    ld_library_path = os.path.join(ogre_prefix, 'lib')
    if os.environ.get('LD_LIBRARY_PATH'):
        ld_library_path += ':' + os.environ['LD_LIBRARY_PATH']
    multiarch = sysconfig.get_config_var('MULTIARCH') or 'aarch64-linux-gnu'
    ogre_plugin_dir = f'/usr/lib/{multiarch}/OGRE-1.9.0'

    bridge_topics = [
        f'/table_setting/joint/{name}@std_msgs/msg/Float64]gz.msgs.Double'
        for name in (
            'base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch',
            'gripper_roll',
            'left_finger', 'right_finger'
        )
    ]
    for name in (
        'glass', 'wine_glass', 'plate', 'cup', 'spoon', 'fork', 'knife',
        'bottle', 'bowl', 'napkin',
    ):
        bridge_topics.extend([
            f'/table_setting/grip/{name}/attach@std_msgs/msg/Empty]gz.msgs.Empty',
            f'/table_setting/grip/{name}/detach@std_msgs/msg/Empty]gz.msgs.Empty',
            f'/table_setting/source_hold/{name}/attach@std_msgs/msg/Empty]gz.msgs.Empty',
            f'/table_setting/source_hold/{name}/detach@std_msgs/msg/Empty]gz.msgs.Empty',
            f'/table_setting/destination_hold/{name}/attach@std_msgs/msg/Empty]gz.msgs.Empty',
            f'/table_setting/destination_hold/{name}/detach@std_msgs/msg/Empty]gz.msgs.Empty',
        ])

    return LaunchDescription([
        DeclareLaunchArgument('meal', default_value='none',
                              description=(
                                  'none, breakfast, lunch, dinner, clean table, '
                                  'or a comma-separated object list')),
        DeclareLaunchArgument('headless', default_value='false',
                              description='false opens Gazebo; true is for automated tests only'),
        DeclareLaunchArgument('speed_scale', default_value='1.0'),
        DeclareLaunchArgument(
            'show_camera', default_value='true',
            description='true opens the annotated computer-vision camera'),
        SetEnvironmentVariable('GZ_RENDERING_ENGINE_PATH', engine_path),
        SetEnvironmentVariable('GZ_RENDERING_RESOURCE_PATH', package_dir),
        SetEnvironmentVariable('LD_LIBRARY_PATH', ld_library_path),
        SetEnvironmentVariable('OGRE_PLUGIN_DIR', ogre_plugin_dir),
        # Gazebo Harmonic + OGRE cannot reliably create its Qt/OpenGL context
        # through Wayland in virtual machines. Force the supported XWayland
        # path and disable MIT-SHM for the Parallels virtual display.
        SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb'),
        SetEnvironmentVariable('QT_X11_NO_MITSHM', '1'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ros_gz_dir + '/launch/gz_sim.launch.py'),
            launch_arguments={'gz_args': gz_args}.items(),
        ),
        TimerAction(
            period=4.0,
            condition=UnlessCondition(headless),
            actions=[ExecuteProcess(
                cmd=['gz', 'sim', '-g', '-v', '3',
                     '--render-engine', 'ogre'],
                output='screen',
            )],
        ),
        # DetachableJoint in Gazebo Harmonic always starts attached. Keep the
        # physics paused while the controller sends its repeated detach pulses,
        # then start the world through Gazebo Transport.
        TimerAction(
            period=3.0,
            actions=[ExecuteProcess(
                cmd=[
                    'gz', 'service',
                    '-s', '/world/arm_table_setting/control',
                    '--reqtype', 'gz.msgs.WorldControl',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '3000',
                    '--req', 'pause: false',
                ],
                output='screen',
            )],
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='table_setting_bridge',
            arguments=bridge_topics,
            output='screen',
        ),
        Node(
            package='ros_gz_image',
            executable='image_bridge',
            name='table_setting_source_rgb_bridge',
            arguments=['/table_setting/camera/image'],
            parameters=[{'qos': 'sensor_data'}],
            output='screen',
        ),
        Node(
            package='ros_gz_image',
            executable='image_bridge',
            name='table_setting_destination_rgb_bridge',
            arguments=['/table_setting/destination_camera/image'],
            parameters=[{'qos': 'sensor_data'}],
            output='screen',
        ),
        Node(
            package='cv_arm_table_setting_demo',
            executable='perception',
            name='source_semantic_perception',
            parameters=[{'scene_config': config, 'camera_role': 'source'}],
            output='screen',
        ),
        Node(
            package='cv_arm_table_setting_demo',
            executable='perception',
            name='destination_semantic_perception',
            parameters=[{'scene_config': config, 'camera_role': 'destination'}],
            output='screen',
        ),
        TimerAction(
            period=7.0,
            actions=[Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='table_setting_destination_cv_view',
                arguments=['/table_setting/vision/destination_debug_image'],
                condition=IfCondition(show_camera),
                output='screen',
            )],
        ),
        TimerAction(
            period=6.0,
            actions=[Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='table_setting_cv_view',
                arguments=['/table_setting/vision/debug_image'],
                condition=IfCondition(show_camera),
                output='screen',
            )],
        ),
        Node(
            package='cv_arm_table_setting_demo',
            executable='controller',
            name='arm_table_setting_controller',
            parameters=[{
                'scene_config': config,
                'meal': meal,
                'speed_scale': speed_scale,
            }],
            output='screen',
        ),
    ])
