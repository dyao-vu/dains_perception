"""
Launch file for GroundingDINO ROS2 node.

Usage:
    ros2 launch groundingdino_ros groundingdino.launch.py
    ros2 launch groundingdino_ros groundingdino.launch.py tracker_type:=clip
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    tracker_type_arg = DeclareLaunchArgument(
        'tracker_type',
        default_value='bytetrack',
        description='Tracker type: bytetrack or clip'
    )

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/viaduct/Sim/SceneDroneSensors/robots/Drone1/sensors/front_center1/scene_camera/image',
        description='Camera image topic to subscribe to'
    )

    output_visualization_arg = DeclareLaunchArgument(
        'output_visualization',
        default_value='true',
        description='Whether to publish visualization images'
    )

    # Get package share directory
    pkg_share = FindPackageShare('groundingdino_ros')

    # Default parameters file path
    default_params_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'default_params.yaml'
    ])

    # GroundingDINO node
    groundingdino_node = Node(
        package='groundingdino_ros',
        executable='groundingdino_node',
        name='groundingdino_node',
        output='screen',
        parameters=[
            default_params_file,
            {
                'tracker_type': LaunchConfiguration('tracker_type'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'output_visualization': LaunchConfiguration('output_visualization'),
            }
        ],
        remappings=[
            ('camera/image', LaunchConfiguration('camera_topic')),
        ]
    )

    return LaunchDescription([
        tracker_type_arg,
        camera_topic_arg,
        output_visualization_arg,
        groundingdino_node,
    ])
