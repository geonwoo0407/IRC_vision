"""Tests for robot-center calibration in the line image."""

import pytest

from step.yolo_line_analyzer import calibrated_robot_center_x
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
