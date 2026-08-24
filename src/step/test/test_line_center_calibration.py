"""Tests for robot-center calibration in the line image."""

import pytest

from step.yolo_line_analyzer import calibrated_robot_center_x


def test_1280_image_center_is_shifted_70_pixels_right():
    center_x = calibrated_robot_center_x(1280, 70.0)

    assert center_x == pytest.approx(710.0)
    assert 710.0 - center_x == pytest.approx(0.0)


def test_center_calibration_is_clipped_inside_image():
    assert calibrated_robot_center_x(1280, -1000.0) == 0.0
    assert calibrated_robot_center_x(1280, 1000.0) == 1279.0
