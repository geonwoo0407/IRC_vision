"""Unit tests for hardware-independent line navigation decisions."""

import pytest

from step.line_navigation_planner import LineNavigationPlanner
from step.line_navigation_planner import NavigationConfig


def line_info(**overrides):
    """Create a valid line analysis sample with optional changes."""
    sample = {
        "detected": True,
        "filtered_heading_error_deg": 0.0,
        "filtered_lateral_offset_norm": 0.0,
        "turn_angle_deg": 0.0,
        "turn_consistency": 0.9,
        "heading_quality": 0.9,
        "geometry_quality": 0.8,
        "detection_quality": 0.85,
    }
    sample.update(overrides)
    return sample


def test_straight_command_contains_speed_and_distance():
    planner = LineNavigationPlanner()

    command = planner.plan(line_info(), 0.1)

    assert command.valid is True
    assert command.motion == "STRAIGHT"
    assert command.linear_speed_mps > 0.0
    assert command.travel_distance_m == pytest.approx(
        command.linear_speed_mps * command.command_duration_sec
    )


@pytest.mark.parametrize(
    ("heading", "expected"),
    [(14.0, "RIGHT"), (-14.0, "LEFT")],
)
def test_heading_sign_selects_turn_direction(heading, expected):
    planner = LineNavigationPlanner()

    for _ in range(3):
        command = planner.plan(
            line_info(
                filtered_heading_error_deg=heading,
                turn_angle_deg=heading * 2.0,
                turn_consistency=0.9,
            ),
            0.1,
        )

    assert command.motion == expected


@pytest.mark.parametrize("heading", [-4.9, 0.0, 4.9])
def test_wider_straight_deadband(heading):
    planner = LineNavigationPlanner()

    command = planner.plan(
        line_info(filtered_heading_error_deg=heading),
        0.1,
    )

    assert command.motion == "STRAIGHT"


def test_far_curve_slows_down_without_starting_turn_early():
    planner = LineNavigationPlanner(
        NavigationConfig(direction_confirmation_frames=1)
    )

    command = planner.plan(
        line_info(
            filtered_lateral_offset_norm=0.1,
            turn_angle_deg=80.0,
        ),
        0.1,
    )

    assert command.motion == "STRAIGHT"
    assert command.reason == "turn_approach_pending"
    assert command.steering_error_deg > 12.0
    assert command.linear_speed_mps < planner.config.max_linear_speed_mps


def test_far_curve_turn_starts_after_near_heading_reaches_corner():
    planner = LineNavigationPlanner(
        NavigationConfig(direction_confirmation_frames=1)
    )

    command = planner.plan(
        line_info(
            filtered_heading_error_deg=6.0,
            filtered_lateral_offset_norm=0.1,
            turn_angle_deg=80.0,
        ),
        0.1,
    )

    assert command.motion == "RIGHT"


def test_conflicting_heading_and_preview_holds_slow_straight():
    planner = LineNavigationPlanner()

    command = planner.plan(
        line_info(
            filtered_heading_error_deg=35.5,
            filtered_lateral_offset_norm=0.049,
            turn_angle_deg=-49.4,
        ),
        0.1,
    )

    assert command.motion == "STRAIGHT"
    assert command.reason == "conflicting_heading_and_preview"
    assert command.steering_error_deg == 0.0
    assert command.linear_speed_mps == planner.config.min_linear_speed_mps


def test_turn_requires_three_consecutive_frames():
    planner = LineNavigationPlanner()
    sample = line_info(filtered_heading_error_deg=20.0)

    commands = [planner.plan(sample, 0.1) for _ in range(3)]

    assert [command.motion for command in commands] == [
        "STRAIGHT",
        "STRAIGHT",
        "RIGHT",
    ]


@pytest.mark.parametrize(
    ("offset", "expected", "lateral_sign"),
    [
        (0.55, "RECOVER_RIGHT", 1),
        (-0.55, "RECOVER_LEFT", -1),
    ],
)
def test_large_offset_creates_separate_recovery_command(
    offset,
    expected,
    lateral_sign,
):
    planner = LineNavigationPlanner()

    command = planner.plan(
        line_info(filtered_lateral_offset_norm=offset),
        0.1,
    )

    assert command.motion == expected
    assert command.reason == "line_center_recovery"
    assert command.linear_speed_mps == 0.0
    assert command.lateral_speed_mps * lateral_sign > 0.0
    assert command.lateral_travel_distance_m * lateral_sign > 0.0


@pytest.mark.parametrize(
    ("offset", "heading", "expected", "angular_sign"),
    [
        (0.593, -32.0, "RECOVER_RIGHT_TURN_LEFT", -1),
        (0.366, 9.8, "RECOVER_RIGHT_TURN_RIGHT", 1),
        (-0.40, -10.0, "RECOVER_LEFT_TURN_LEFT", -1),
        (-0.40, 10.0, "RECOVER_LEFT_TURN_RIGHT", 1),
    ],
)
def test_recovery_separates_line_side_from_turn_direction(
    offset,
    heading,
    expected,
    angular_sign,
):
    planner = LineNavigationPlanner()

    command = planner.plan(
        line_info(
            filtered_lateral_offset_norm=offset,
            filtered_heading_error_deg=heading,
        ),
        0.1,
    )

    assert command.motion == expected
    assert command.angular_speed_rad_s * angular_sign > 0.0


def test_straight_line_heading_error_uses_recovery_not_plain_left():
    planner = LineNavigationPlanner()

    command = planner.plan(
        line_info(
            filtered_lateral_offset_norm=-0.10,
            filtered_heading_error_deg=-20.0,
            turn_angle_deg=None,
            turn_consistency=None,
        ),
        0.1,
    )

    assert command.motion == "RECOVER_LEFT_TURN_LEFT"


def test_plain_left_is_reserved_for_confirmed_left_curve():
    planner = LineNavigationPlanner()
    curve = line_info(
        filtered_lateral_offset_norm=-0.10,
        filtered_heading_error_deg=-20.0,
        turn_angle_deg=-30.0,
        turn_consistency=0.9,
    )

    commands = [planner.plan(curve, 0.1) for _ in range(3)]

    assert commands[-1].motion == "LEFT"


def test_moderate_offset_and_parallel_line_can_continue_straight():
    planner = LineNavigationPlanner()

    command = planner.plan(
        line_info(
            filtered_lateral_offset_norm=-0.40,
            filtered_heading_error_deg=-0.2,
            turn_angle_deg=-2.3,
        ),
        0.1,
    )

    assert command.motion == "STRAIGHT"


def test_moderate_offset_still_recovers_when_heading_is_too_large():
    planner = LineNavigationPlanner()

    command = planner.plan(
        line_info(
            filtered_lateral_offset_norm=-0.383,
            filtered_heading_error_deg=-9.4,
            turn_angle_deg=None,
        ),
        0.1,
    )

    assert command.motion == "RECOVER_LEFT_TURN_LEFT"


def test_recovery_uses_exit_threshold_before_returning_to_tracking():
    planner = LineNavigationPlanner()

    first = planner.plan(
        line_info(filtered_lateral_offset_norm=0.55),
        0.1,
    )
    held = planner.plan(
        line_info(filtered_lateral_offset_norm=0.50),
        0.1,
    )
    released = planner.plan(
        line_info(filtered_lateral_offset_norm=0.10),
        0.1,
    )

    assert first.motion == "RECOVER_RIGHT"
    assert held.motion == "RECOVER_RIGHT"
    assert released.motion == "STRAIGHT"


def test_small_or_inconsistent_far_curve_is_not_used_early():
    planner = LineNavigationPlanner()

    small_curve = planner.plan(
        line_info(turn_angle_deg=7.9, turn_consistency=1.0),
        0.1,
    )
    inconsistent_curve = LineNavigationPlanner().plan(
        line_info(turn_angle_deg=30.0, turn_consistency=0.2),
        0.1,
    )

    assert small_curve.preview_component_deg == 0.0
    assert inconsistent_curve.preview_component_deg == 0.0


def test_angular_acceleration_is_limited():
    config = NavigationConfig(max_angular_accel_rad_s2=0.5)
    planner = LineNavigationPlanner(config)

    command = planner.plan(
        line_info(
            filtered_heading_error_deg=40.0,
            turn_angle_deg=40.0,
            turn_consistency=0.9,
        ),
        0.1,
    )

    assert command.angular_accel_rad_s2 == pytest.approx(0.5)
    assert command.angular_speed_rad_s == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("sample", "reason"),
    [
        ({"detected": False}, "line_not_detected"),
        (
            line_info(geometry_quality=0.1),
            "low_line_quality",
        ),
    ],
)
def test_unsafe_input_produces_stop(sample, reason):
    planner = LineNavigationPlanner()

    command = planner.plan(sample, 0.1)

    assert command.valid is False
    assert command.motion == "STOP"
    assert command.reason == reason
    assert command.linear_speed_mps == 0.0
