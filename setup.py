from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'cv_arm_table_setting_demo'


def install_tree(source, destination):
    """Install a resource directory while preserving its layout."""
    entries = []
    for directory, _, filenames in os.walk(source):
        if not filenames:
            continue
        relative = os.path.relpath(directory, source)
        target = destination if relative == '.' else os.path.join(destination, relative)
        entries.append((target, [os.path.join(directory, name) for name in filenames]))
    return entries

setup(
    name=package_name,
    version='1.3.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ] + install_tree('ogre', os.path.join('share', package_name, 'ogre')),
    install_requires=['setuptools', 'PyYAML', 'numpy'],
    zip_safe=True,
    maintainer='Table Setting Project',
    maintainer_email='student@example.com',
    description='AI visual recognition and articulated-arm table setting in Gazebo.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'perception = cv_arm_table_setting_demo.perception_node:main',
            'controller = cv_arm_table_setting_demo.controller_node:main',
        ],
    },
)
