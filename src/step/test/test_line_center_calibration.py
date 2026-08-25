"""Tests for robot-center calibration in the line image."""

import pytest

from step.yolo_line_analyzer import calibrated_robot_center_x
from step.yolo_line_analyzer import ground_forward_distance_from_depth
from step.yolo_line_analyzer import LinePoint
from step.yolo_line_analyzer import YoloLineAnalyzer
from step.temporal_confirmation import TemporalConfirmationFilter


def test_1280_image_center_is_shifted_70_pixels_right():
    center_x = calibrated_robot_center_x(1280, 70.0)

    assert center_x == pytest.approx(710.0)
    assert 710.0 - center_x == pytest.approx(0.0)


def test_center_calibration_is_clipped_inside_image():
    assert calibrated_robot_center_x(1280, -1000.0) == 0.0
    assert calibrated_robot_center_x(1280, 1000.0) == 1279.0


def test_line_roi_uses_fifteen_to_eighty_five_percent_width():
    analyzer = YoloLineAnalyzer.__new__(YoloLineAnalyzer)
    analyzer.image_width = 1280
    analyzer.image_height = 720
    analyzer.roi_x_min_ratio = 0.15
    analyzer.roi_x_max_ratio = 0.85
    analyzer.roi_y_min_ratio = 0.0
    analyzer.roi_y_max_ratio = 1.0
    analyzer.line_class_name = "line"
    analyzer.min_confidence = 0.4
    analyzer.min_bbox_width_px = 3
    analyzer.min_bbox_height_px = 3
    analyzer.max_points = 30

    def detection(center_x):
        return {
            "class_name": "line",
            "confidence": 0.9,
            "bbox": [center_x - 2, 358, center_x + 2, 362],
            "center": [center_x, 360],
        }

    points = analyzer._extract_line_points(
        [
            detection(191),
            detection(192),
            detection(1088),
            detection(1089),
        ]
    )

    assert [point.x for point in points] == [192.0, 1088.0]


def test_path_turn_delta_separates_straight_and_right_curve():
    straight = YoloLineAnalyzer._calculate_angle_change_statistics(
        [12.0, 12.2, 12.1]
    )
    right_curve = YoloLineAnalyzer._calculate_angle_change_statistics(
        [10.0, 16.0, 24.0]
    )

    assert straight["path_turn_delta_deg"] == pytest.approx(0.1)
    assert right_curve["path_turn_delta_deg"] == pytest.approx(14.0)
    assert right_curve["turn_consistency"] == pytest.approx(1.0)


def corner_geometry(points):
    """Run corner geometry with the production safety thresholds."""
    return YoloLineAnalyzer._detect_corner_start_geometry(
        points,
        min_points=3,
        min_segment_length_px=20.0,
        straight_max_turn_delta_deg=15.0,
        min_turn_delta_deg=30.0,
        onset_deviation_deg=15.0,
        min_consistent_segments=1,
        min_consistency=0.75,
    )


def test_real_right_corner_finds_entry_one_point_before_confirmed_bend():
    points = [
        LinePoint(500.0, 700.0, 0.9),
        LinePoint(500.0, 630.0, 0.9),
        LinePoint(500.0, 560.0, 0.9),
        LinePoint(510.0, 490.0, 0.9),
        LinePoint(550.0, 420.0, 0.9),
        LinePoint(620.0, 350.0, 0.9),
        LinePoint(710.0, 280.0, 0.9),
    ]

    result = corner_geometry(points)

    assert result["detected"] is True
    assert result["direction"] == "RIGHT"
    assert result["start_index"] == 3
    assert result["start_point"] == points[3]
    assert result["turn_delta_deg"] > 30.0
    assert result["consistency"] == pytest.approx(1.0)


def test_real_left_corner_is_symmetric():
    points = [
        LinePoint(500.0, 700.0, 0.9),
        LinePoint(500.0, 630.0, 0.9),
        LinePoint(500.0, 560.0, 0.9),
        LinePoint(490.0, 490.0, 0.9),
        LinePoint(450.0, 420.0, 0.9),
        LinePoint(380.0, 350.0, 0.9),
        LinePoint(290.0, 280.0, 0.9),
    ]

    result = corner_geometry(points)

    assert result["detected"] is True
    assert result["direction"] == "LEFT"
    assert result["start_index"] == 3


def test_slanted_straight_line_never_creates_corner_preview():
    points = [
        LinePoint(500.0 + index * 20.0, 700.0 - index * 70.0, 0.9)
        for index in range(7)
    ]

    result = corner_geometry(points)

    assert result["detected"] is False
    assert result["state"] == "STRAIGHT"
    assert result["direction"] is None


@pytest.mark.parametrize(
    ("far_x", "expected_direction"),
    [(600.0, "RIGHT"), (400.0, "LEFT")],
)
def test_three_points_are_enough_to_find_corner_at_middle_point(
    far_x,
    expected_direction,
):
    points = [
        LinePoint(500.0, 700.0, 0.9),
        LinePoint(500.0, 600.0, 0.9),
        LinePoint(far_x, 600.0, 0.9),
    ]

    result = corner_geometry(points)

    assert result["detected"] is True
    assert result["direction"] == expected_direction
    assert result["start_index"] == 1
    assert result["start_point"] == points[1]


def test_twenty_degree_bend_is_ambiguous_instead_of_corner():
    points = [
        LinePoint(500.0, 700.0, 0.9),
        LinePoint(500.0, 600.0, 0.9),
        LinePoint(536.4, 500.0, 0.9),
    ]

    result = corner_geometry(points)

    assert result["detected"] is False
    assert result["state"] == "AMBIGUOUS"


def test_floor_forward_projection_removes_camera_pitch_slant():
    pitch_deg = 45.0
    robot_forward_m = 0.50
    camera_height_m = 0.70
    depth_m = (robot_forward_m + camera_height_m) / (2.0 ** 0.5)
    camera_down_m = (
        -robot_forward_m + camera_height_m
    ) / (2.0 ** 0.5)
    fy = 1000.0
    cy = 360.0
    y_px = cy + fy * camera_down_m / depth_m

    lateral_m, forward_m = ground_forward_distance_from_depth(
        x_px=640.0,
        y_px=y_px,
        depth_m=depth_m,
        fx=1000.0,
        fy=fy,
        cx=640.0,
        cy=cy,
        camera_pitch_down_deg=pitch_deg,
        camera_forward_offset_m=0.0,
    )

    assert lateral_m == pytest.approx(0.0)
    assert forward_m == pytest.approx(robot_forward_m)


def test_corner_requires_three_depth_valid_frames_and_counts_motions():
    analyzer = YoloLineAnalyzer.__new__(YoloLineAnalyzer)
    analyzer.corner_min_points = 3
    analyzer.corner_min_segment_length_px = 20.0
    analyzer.corner_straight_max_turn_delta_deg = 15.0
    analyzer.corner_min_turn_delta_deg = 30.0
    analyzer.corner_onset_deviation_deg = 15.0
    analyzer.corner_min_consistent_segments = 1
    analyzer.corner_min_consistency = 0.75
    analyzer.corner_turn_margin_m = 0.15
    analyzer.corner_straight_motion_distance_m = 0.05
    analyzer.corner_confirmation_filter = TemporalConfirmationFilter(
        window_size=5,
        required_hits=3,
        max_missed_frames=5,
        spatial_matching=False,
    )
    analyzer.corner_candidate_direction = None
    analyzer.corner_hold_sec = 0.30

    def measured_distance(_point):
        return {
            "point_px": [500, 560],
            "depth_m": 0.82,
            "lateral_offset_m": -0.1,
            "ground_forward_distance_m": 0.826,
            "horizontal_distance_m": 0.826,
            "depth_valid": True,
        }

    analyzer._line_point_distance = measured_distance
    points = [
        LinePoint(500.0, 700.0, 0.9),
        LinePoint(500.0, 630.0, 0.9),
        LinePoint(500.0, 560.0, 0.9),
        LinePoint(510.0, 490.0, 0.9),
        LinePoint(550.0, 420.0, 0.9),
        LinePoint(620.0, 350.0, 0.9),
        LinePoint(710.0, 280.0, 0.9),
    ]

    previews = [
        analyzer._analyze_corner_preview(points)
        for _ in range(3)
    ]

    assert previews[0]["corner_preview_confirmed"] is False
    assert previews[1]["corner_preview_confirmed"] is False
    assert previews[2]["corner_preview_confirmed"] is True
    assert previews[2]["corner_direction"] == "RIGHT"
    assert previews[2]["corner_straight_motion_count"] == 13


def test_corner_confirms_three_of_five_and_holds_brief_miss(monkeypatch):
    analyzer = YoloLineAnalyzer.__new__(YoloLineAnalyzer)
    analyzer.corner_min_points = 3
    analyzer.corner_min_segment_length_px = 20.0
    analyzer.corner_straight_max_turn_delta_deg = 15.0
    analyzer.corner_min_turn_delta_deg = 30.0
    analyzer.corner_onset_deviation_deg = 15.0
    analyzer.corner_min_consistent_segments = 1
    analyzer.corner_min_consistency = 0.75
    analyzer.corner_turn_margin_m = 0.15
    analyzer.corner_straight_motion_distance_m = 0.05
    analyzer.corner_confirmation_filter = TemporalConfirmationFilter(
        window_size=5,
        required_hits=3,
        max_missed_frames=5,
        spatial_matching=False,
    )
    analyzer.corner_candidate_direction = None
    analyzer.corner_hold_sec = 0.30

    def measured_distance(_point):
        return {
            "point_px": [500, 600],
            "depth_m": 0.70,
            "lateral_offset_m": 0.0,
            "ground_forward_distance_m": 0.50,
            "horizontal_distance_m": 0.50,
            "depth_valid": True,
        }

    analyzer._line_point_distance = measured_distance
    corner = [
        LinePoint(500.0, 700.0, 0.9),
        LinePoint(500.0, 600.0, 0.9),
        LinePoint(600.0, 600.0, 0.9),
    ]
    ambiguous = [
        LinePoint(500.0, 700.0, 0.9),
        LinePoint(500.0, 600.0, 0.9),
        LinePoint(536.4, 500.0, 0.9),
    ]
    clock = [0.0]
    monkeypatch.setattr(
        "step.yolo_line_analyzer.time.monotonic",
        lambda: clock[0],
    )

    first = analyzer._analyze_corner_preview(corner)
    clock[0] = 0.05
    miss = analyzer._analyze_corner_preview(ambiguous)
    clock[0] = 0.10
    second = analyzer._analyze_corner_preview(corner)
    clock[0] = 0.15
    confirmed = analyzer._analyze_corner_preview(corner)
    clock[0] = 0.25
    held = analyzer._analyze_corner_preview(ambiguous)
    clock[0] = 0.50
    expired = analyzer._analyze_corner_preview(ambiguous)

    assert first["corner_preview_confirmed"] is False
    assert miss["corner_preview_confirmed"] is False
    assert second["corner_preview_confirmed"] is False
    assert confirmed["corner_preview_confirmed"] is True
    assert held["corner_preview_confirmed"] is True
    assert held["corner_preview_held"] is True
    assert expired["corner_preview_confirmed"] is False
