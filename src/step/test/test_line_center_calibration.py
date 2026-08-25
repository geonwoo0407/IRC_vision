"""Tests for robot-center calibration in the line image."""

import pytest

from step.yolo_line_analyzer import calibrated_robot_center_x
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
        min_points=6,
        min_segment_dy_px=20.0,
        min_turn_delta_deg=45.0,
        onset_deviation_deg=12.0,
        min_consistent_segments=2,
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
    assert result["start_index"] == 2
    assert result["start_point"] == points[2]
    assert result["turn_delta_deg"] > 45.0
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
    assert result["start_index"] == 2


def test_slanted_straight_line_never_creates_corner_preview():
    points = [
        LinePoint(500.0 + index * 20.0, 700.0 - index * 70.0, 0.9)
        for index in range(7)
    ]

    result = corner_geometry(points)

    assert result["detected"] is False
    assert result["direction"] is None


def test_corner_requires_three_depth_valid_frames_and_counts_motions():
    analyzer = YoloLineAnalyzer.__new__(YoloLineAnalyzer)
    analyzer.corner_min_points = 6
    analyzer.min_segment_dy_px = 20.0
    analyzer.corner_min_turn_delta_deg = 45.0
    analyzer.corner_onset_deviation_deg = 12.0
    analyzer.corner_min_consistent_segments = 2
    analyzer.corner_min_consistency = 0.75
    analyzer.corner_turn_margin_m = 0.15
    analyzer.corner_straight_motion_distance_m = 0.05
    analyzer.corner_confirmation_filter = TemporalConfirmationFilter(
        window_size=3,
        required_hits=3,
        max_missed_frames=1,
        spatial_matching=False,
    )
    analyzer.corner_candidate_direction = None

    def measured_distance(_point):
        return {
            "point_px": [500, 560],
            "depth_m": 0.82,
            "lateral_offset_m": -0.1,
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
