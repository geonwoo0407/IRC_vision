#!/usr/bin/env python3
"""Launch camera, detector, and analyzers without a motion planner."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build a Vision-owned graph that publishes observations only."""
    enable_camera = LaunchConfiguration("enable_camera")
    device = LaunchConfiguration("device")
    display = LaunchConfiguration("display")
    metrics_mode = LaunchConfiguration("metrics_mode")
    max_fps = LaunchConfiguration("max_fps")
    camera_topic_prefix = LaunchConfiguration("camera_topic_prefix")
    publish_annotated_image = LaunchConfiguration("publish_annotated_image")
    robot_center_offset_px = LaunchConfiguration("robot_center_offset_px")
    camera_pitch_down_deg = LaunchConfiguration("camera_pitch_down_deg")
    camera_forward_offset_m = LaunchConfiguration("camera_forward_offset_m")

    color_topic = PythonExpression(
        ["'", camera_topic_prefix, "/color/image_raw'"]
    )
    depth_topic = PythonExpression(
        ["'", camera_topic_prefix, "/aligned_depth_to_color/image_raw'"]
    )
    camera_info_topic = PythonExpression(
        ["'", camera_topic_prefix, "/color/camera_info'"]
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("realsense2_camera"),
                    "launch",
                    "rs_launch.py",
                ]
            )
        ),
        condition=IfCondition(enable_camera),
        launch_arguments={
            "align_depth.enable": "true",
            "enable_gyro": "false",
            "enable_accel": "false",
            "rgb_camera.color_profile": "1280x720x30",
            "depth_module.depth_profile": "848x480x30",
        }.items(),
    )

    detector = Node(
        package="step",
        executable="yolo26_detector",
        name="yolo26_detector",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "device": device,
                "display": ParameterValue(display, value_type=bool),
                "metrics_mode": metrics_mode,
                "max_fps": ParameterValue(max_fps, value_type=float),
                "image_topic": ParameterValue(color_topic, value_type=str),
                "publish_annotated_image": ParameterValue(
                    publish_annotated_image,
                    value_type=bool,
                ),
            }
        ],
    )

    analyzers = Node(
        package="step",
        executable="unified_vision_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "image_topic": ParameterValue(color_topic, value_type=str),
                "depth_topic": ParameterValue(depth_topic, value_type=str),
                "camera_info_topic": ParameterValue(
                    camera_info_topic,
                    value_type=str,
                ),
                "robot_center_offset_px": ParameterValue(
                    robot_center_offset_px,
                    value_type=float,
                ),
                "camera_pitch_down_deg": ParameterValue(
                    camera_pitch_down_deg,
                    value_type=float,
                ),
                "camera_forward_offset_m": ParameterValue(
                    camera_forward_offset_m,
                    value_type=float,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_camera", default_value="true"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("display", default_value="true"),
            DeclareLaunchArgument("metrics_mode", default_value="auto"),
            DeclareLaunchArgument("max_fps", default_value="30.0"),
            DeclareLaunchArgument(
                "camera_topic_prefix",
                default_value=EnvironmentVariable(
                    "IRC_CAMERA_TOPIC_PREFIX",
                    default_value="/camera/camera",
                ),
            ),
            DeclareLaunchArgument(
                "publish_annotated_image",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "robot_center_offset_px",
                default_value="70.0",
            ),
            DeclareLaunchArgument(
                "camera_pitch_down_deg",
                default_value="45.0",
            ),
            DeclareLaunchArgument(
                "camera_forward_offset_m",
                default_value="0.0",
            ),
            camera,
            detector,
            analyzers,
        ]
    )
