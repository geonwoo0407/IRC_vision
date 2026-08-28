#!/usr/bin/env python3
"""Launch the complete STEP vision and motion-decision pipeline."""

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
    """Build the complete camera-to-navigation ROS graph."""
    enable_camera = LaunchConfiguration("enable_camera")
    device = LaunchConfiguration("device")
    display = LaunchConfiguration("display")
    metrics_mode = LaunchConfiguration("metrics_mode")
    max_fps = LaunchConfiguration("max_fps")
    camera_topic_prefix = LaunchConfiguration("camera_topic_prefix")

    color_image_topic = PythonExpression(
        ["'", camera_topic_prefix, "/color/image_raw'"]
    )
    aligned_depth_topic = PythonExpression(
        ["'", camera_topic_prefix, "/aligned_depth_to_color/image_raw'"]
    )
    color_camera_info_topic = PythonExpression(
        ["'", camera_topic_prefix, "/color/camera_info'"]
    )

    publish_annotated_image = LaunchConfiguration(
        "publish_annotated_image"
    )
    initial_mission_phase = LaunchConfiguration("initial_mission_phase")
    enable_ball_lost_recovery = LaunchConfiguration(
        "enable_ball_lost_recovery"
    )
    recovery_heading_turn_deg = LaunchConfiguration(
        "recovery_heading_turn_deg"
    )
    recovery_away_heading_turn_deg = LaunchConfiguration(
        "recovery_away_heading_turn_deg"
    )
    curve_follow_max_offset_norm = LaunchConfiguration(
        "curve_follow_max_offset_norm"
    )
    robot_center_offset_px = LaunchConfiguration(
        "robot_center_offset_px"
    )
    camera_pitch_down_deg = LaunchConfiguration(
        "camera_pitch_down_deg"
    )
    camera_forward_offset_m = LaunchConfiguration(
        "camera_forward_offset_m"
    )
    line_roi_x_min_ratio = LaunchConfiguration(
        "line_roi_x_min_ratio"
    )
    line_roi_x_max_ratio = LaunchConfiguration(
        "line_roi_x_max_ratio"
    )
    corner_min_turn_delta_deg = LaunchConfiguration(
        "corner_min_turn_delta_deg"
    )
    corner_straight_max_turn_delta_deg = LaunchConfiguration(
        "corner_straight_max_turn_delta_deg"
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
                "image_topic": ParameterValue(
                    color_image_topic,
                    value_type=str,
                ),
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
                "image_topic": ParameterValue(
                    color_image_topic,
                    value_type=str,
                ),
                "depth_topic": ParameterValue(
                    aligned_depth_topic,
                    value_type=str,
                ),
                "camera_info_topic": ParameterValue(
                    color_camera_info_topic,
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
                "roi_x_min_ratio": ParameterValue(
                    line_roi_x_min_ratio,
                    value_type=float,
                ),
                "roi_x_max_ratio": ParameterValue(
                    line_roi_x_max_ratio,
                    value_type=float,
                ),
                "corner_min_turn_delta_deg": ParameterValue(
                    corner_min_turn_delta_deg,
                    value_type=float,
                ),
                "corner_straight_max_turn_delta_deg": ParameterValue(
                    corner_straight_max_turn_delta_deg,
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
                "enable_ball_lost_recovery": ParameterValue(
                    enable_ball_lost_recovery,
                    value_type=bool,
                ),
                "recovery_heading_turn_deg": ParameterValue(
                    recovery_heading_turn_deg,
                    value_type=float,
                ),
                "recovery_away_heading_turn_deg": ParameterValue(
                    recovery_away_heading_turn_deg,
                    value_type=float,
                ),
                "curve_follow_max_offset_norm": ParameterValue(
                    curve_follow_max_offset_norm,
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
                "camera_topic_prefix",
                default_value=EnvironmentVariable(
                    "IRC_CAMERA_TOPIC_PREFIX",
                    default_value="/camera/camera",
                ),
                description=(
                    "RealSense topic prefix. "
                    "PC=/camera, Jetson=/camera/camera."
                ),
            ),
            DeclareLaunchArgument(
                "publish_annotated_image",
                default_value="false",
                description="Publish the annotated YOLO image ROS topic.",
            ),
            DeclareLaunchArgument(
                "initial_mission_phase",
                default_value="AUTO",
                description="Initial planner phase before /mission/phase is received.",
            ),
            DeclareLaunchArgument(
                "enable_ball_lost_recovery",
                default_value="true",
                description=(
                    "Keep ball ownership and search from its last pose when "
                    "detection is lost."
                ),
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
                "curve_follow_max_offset_norm",
                default_value="0.55",
                description=(
                    "Maximum offset that a reliable matching curve may "
                    "override before recovery is allowed again."
                ),
            ),
            DeclareLaunchArgument(
                "robot_center_offset_px",
                default_value="70.0",
                description="Robot center shift right from image midpoint.",
            ),
            DeclareLaunchArgument(
                "camera_pitch_down_deg",
                default_value="45.0",
                description="Color camera downward pitch used for floor distance.",
            ),
            DeclareLaunchArgument(
                "camera_forward_offset_m",
                default_value="0.0",
                description=(
                    "Camera forward position from the robot distance origin."
                ),
            ),
            DeclareLaunchArgument(
                "line_roi_x_min_ratio",
                default_value="0.15",
                description="Left image boundary used by the line analyzer.",
            ),
            DeclareLaunchArgument(
                "line_roi_x_max_ratio",
                default_value="0.85",
                description="Right image boundary used by the line analyzer.",
            ),
            DeclareLaunchArgument(
                "corner_min_turn_delta_deg",
                default_value="30.0",
                description="Minimum path bend for a real corner preview.",
            ),
            DeclareLaunchArgument(
                "corner_straight_max_turn_delta_deg",
                default_value="15.0",
                description="Maximum bend classified as definitely straight.",
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
