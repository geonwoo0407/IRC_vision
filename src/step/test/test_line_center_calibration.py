"""Tests for robot-center calibration in the line image."""

import pytest

from step.yolo_line_analyzer import calibrated_robot_center_x
from step.yolo_line_analyzer import LinePoint
from step.yolo_line_analyzer import YoloLineAnalyzer


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


def target_analyzer():
    """Create a ROS-free analyzer shell for target geometry tests."""
    analyzer = YoloLineAnalyzer.__new__(YoloLineAnalyzer)
    analyzer.image_width = 1280
    analyzer.image_height = 720
    analyzer.robot_center_offset_px = 70.0
    analyzer.recovery_target_y_min_ratio = 0.55
    analyzer.recovery_target_y_max_ratio = 0.75
    analyzer.min_points_for_fit = 2
    analyzer.min_segment_dy_px = 20.0
    analyzer.outlier_residual_px = 50.0
    analyzer.max_fit_iterations = 3
    return analyzer


def test_recovery_target_fits_points_in_55_to_75_percent_band():
    analyzer = target_analyzer()
    points = [
        LinePoint(454.0, 540.0, 0.9),
        LinePoint(447.0, 470.0, 0.9),
        LinePoint(440.0, 400.0, 0.9),
        LinePoint(420.0, 200.0, 0.9),
    ]

    target = analyzer._calculate_recovery_target(points, -10.0)

    assert target["recovery_target_source"] == "band_fit"
    assert target["recovery_target_point_count"] == 3
    assert target["recovery_target_point_px"] == [440, 396]
    assert target["recovery_target_angle_deg"] == pytest.approx(
        -39.86,
        abs=0.05,
    )


def test_recovery_target_uses_single_band_point_when_fit_is_unavailable():
    analyzer = target_analyzer()
    points = [
        LinePoint(500.0, 450.0, 0.9),
        LinePoint(420.0, 200.0, 0.9),
    ]

    target = analyzer._calculate_recovery_target(points, -10.0)

    assert target["recovery_target_source"] == "band_point"
    assert target["recovery_target_point_px"] == [500, 450]
    assert target["recovery_target_angle_deg"] == pytest.approx(
        -37.87,
        abs=0.05,
    )


def test_recovery_target_falls_back_to_heading_without_band_point():
    analyzer = target_analyzer()

    target = analyzer._calculate_recovery_target(
        [LinePoint(420.0, 200.0, 0.9)],
        -10.6,
    )

    assert target["recovery_target_source"] == "heading_fallback"
    assert target["recovery_target_point_px"] is None
    assert target["recovery_target_angle_deg"] == pytest.approx(-10.6)
