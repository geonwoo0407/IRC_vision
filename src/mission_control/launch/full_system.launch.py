#!/usr/bin/env python3
"""Launch the complete STEP vision and motion-decision pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the complete camera-to-navigation ROS graph."""
    enable_camera = LaunchConfiguration("enable_camera")
    device = LaunchConfiguration("device")
    display = LaunchConfiguration("display")
    metrics_mode = LaunchConfiguration("metrics_mode")
    max_fps = LaunchConfiguration("max_fps")
    initial_mission_phase = LaunchConfiguration("initial_mission_phase")
    recovery_heading_turn_deg = LaunchConfiguration(
        "recovery_heading_turn_deg"
    )
    recovery_away_heading_turn_deg = LaunchConfiguration(
        "recovery_away_heading_turn_deg"
    )
    robot_center_offset_px = LaunchConfiguration(
        "robot_center_offset_px"
    )
    corner_min_turn_delta_deg = LaunchConfiguration(
        "corner_min_turn_delta_deg"
    )
    corner_straight_motion_distance_m = LaunchConfiguration(
        "corner_straight_motion_distance_m"
    )
    corner_turn_margin_m = LaunchConfiguration(
        "corner_turn_margin_m"
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        condition=IfCondition(enable_camera),
        launch_arguments={
            "align_depth.enable": "true",
            "enable_gyro": "true",
            "enable_accel": "true",
            "rgb_camera.color_profile": "1280,720,30",
            "depth_module.depth_profile": "848,480,30",
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
                "robot_center_offset_px": ParameterValue(
                    robot_center_offset_px,
                    value_type=float,
                ),
                "corner_min_turn_delta_deg": ParameterValue(
                    corner_min_turn_delta_deg,
                    value_type=float,
                ),
                "corner_straight_motion_distance_m": ParameterValue(
                    corner_straight_motion_distance_m,
                    value_type=float,
                ),
                "corner_turn_margin_m": ParameterValue(
                    corner_turn_margin_m,
                    value_type=float,
                ),
            }
        ],
    )

    motion_decision = Node(
        package="mission_control",
        executable="motion_decision_node",
        name="motion_decision_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "initial_mission_phase": initial_mission_phase,
                "recovery_heading_turn_deg": ParameterValue(
                    recovery_heading_turn_deg,
                    value_type=float,
                ),
                "recovery_away_heading_turn_deg": ParameterValue(
                    recovery_away_heading_turn_deg,
                    value_type=float,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_camera",
                default_value="true",
                description="Launch the RealSense camera and aligned depth stream.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="ONNX Runtime device used by yolo26_detector.",
            ),
            DeclareLaunchArgument(
                "display",
                default_value="true",
                description="Show the detector visualization window.",
            ),
            DeclareLaunchArgument(
                "metrics_mode",
                default_value="auto",
                description="Metrics overlay selected by yolo26_detector.",
            ),
            DeclareLaunchArgument(
                "max_fps",
                default_value="30.0",
                description="Maximum detector processing rate.",
            ),
            DeclareLaunchArgument(
                "initial_mission_phase",
                default_value="AUTO",
                description="Initial planner phase before /mission/phase is received.",
            ),
            DeclareLaunchArgument(
                "recovery_heading_turn_deg",
                default_value="10.0",
                description="Heading deadband before numbered recovery turns.",
            ),
            DeclareLaunchArgument(
                "recovery_away_heading_turn_deg",
                default_value="3.0",
                description=(
                    "Heading deadband when an off-center robot points "
                    "farther away from the line."
                ),
            ),
            DeclareLaunchArgument(
                "robot_center_offset_px",
                default_value="70.0",
                description="Robot center shift right from image midpoint.",
            ),
            DeclareLaunchArgument(
                "corner_min_turn_delta_deg",
                default_value="45.0",
                description="Minimum path bend for a real corner preview.",
            ),
            DeclareLaunchArgument(
                "corner_straight_motion_distance_m",
                default_value="0.05",
                description="Forward distance of one straight walk motion.",
            ),
            DeclareLaunchArgument(
                "corner_turn_margin_m",
                default_value="0.15",
                description="Distance reserved before the corner starts.",
            ),
            camera,
            detector,
            analyzers,
            motion_decision,
        ]
    )
