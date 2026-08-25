"""Unit tests for unified mission command selection."""

import pytest

from mission_control.motion_decision_planner import MotionDecisionConfig
from mission_control.motion_decision_planner import MotionDecisionPlanner


def line_info(**overrides):
    sample = {
        "detected": True,
        "filtered_heading_error_deg": 0.0,
        "filtered_lateral_offset_norm": 0.0,
        "heading_quality": 0.9,
        "geometry_quality": 0.9,
        "detection_quality": 0.9,
        "turn_angle_deg": 0.0,
        "turn_consistency": 1.0,
    }
    sample.update(overrides)
    return sample


def confirmed_right_corner_line():
    """Create line data that would otherwise approach a right corner."""
    return line_info(
        corner_preview_confirmed=True,
        corner_direction="RIGHT",
        corner_start_distance_m=0.61,
        corner_remaining_forward_m=0.46,
        corner_straight_motion_count=9,
        turn_angle_deg=75.0,
        turn_consistency=0.95,
    )


def ball_info(**overrides):
    sample = {
        "detected": True,
        "confidence": 0.9,
        "depth_valid": True,
        "depth_m": 1.2,
        "distance_m": 1.2,
        "bearing_deg": 0.0,
        "offset_x_norm": 0.0,
        "pickup_ready": False,
        "pickup_now": False,
    }
    sample.update(overrides)
    return sample


def goal_info(**overrides):
    sample = {
        "detected": True,
        "confidence": 0.9,
        "depth_valid": True,
        "depth_m": 0.25,
        "distance_m": 0.25,
        "bearing_deg": 0.0,
        "offset_x_norm": 0.0,
    }
    sample.update(overrides)
    return sample


def hurdle_info(**overrides):
    sample = {
        "detected": True,
        "confidence": 0.9,
        "depth_valid": True,
        "depth_m": 0.8,
        "distance_m": 0.8,
        "ground_gap_m": 0.1,
        "camera_bottom_gap_m": 0.02,
        "bearing_deg": 0.0,
        "offset_x_norm": 0.0,
        "hurdle_angle_deg": 0.0,
        "bbox": [300, 300, 980, 500],
        "image_width": 1280,
        "image_height": 720,
    }
    sample.update(overrides)
    return sample


def hurdle_route_line(x=710):
    """Create unoccluded line points below and above a hurdle."""
    sample = line_info()
    sample.update(
        {
            "robot_center_x_px": 710.0,
            "image_width": 1280,
            "image_height": 720,
            "center_points_px": [
                [x, 650],
                [x, 580],
                [x, 250],
                [x, 180],
            ],
        }
    )
    return sample


def observations(**overrides):
    samples = {
        "line": None,
        "ball": None,
        "goal": None,
        "hurdle": None,
    }
    samples.update(overrides)
    return samples


def recovery_planner():
    """Create a planner with the currently disabled legacy recovery enabled."""
    return MotionDecisionPlanner(
        MotionDecisionConfig(enable_ball_lost_recovery=True)
    )


def test_explicit_goal_phase_ignores_visible_ball():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "GOAL_APPROACH",
        observations(ball=ball_info(), goal=goal_info()),
        0.1,
    )

    assert decision.source == "goal"
    assert decision.action == "SHOT"
    assert decision.requires_ack is True


def test_blocking_hurdle_has_priority_over_close_ball():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=0.85, distance_m=0.86),
            goal=goal_info(),
            hurdle=hurdle_info(),
        ),
        0.1,
    )

    assert decision.source == "hurdle"
    assert decision.action == "GO"


@pytest.mark.parametrize(
    ("target_name", "target", "expected_source"),
    [
        (
            "ball",
            ball_info(depth_m=0.85, distance_m=0.85),
            "ball",
        ),
        (
            "goal",
            goal_info(depth_m=0.45, distance_m=0.45),
            "goal",
        ),
        (
            "hurdle",
            hurdle_info(depth_m=0.90, distance_m=0.90),
            "hurdle",
        ),
    ],
)
def test_control_range_mission_preempts_corner_during_line_phase(
    target_name,
    target,
    expected_source,
):
    """An eligible mission target owns control before corner approach."""
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "LINE_TRACK",
        observations(
            line=confirmed_right_corner_line(),
            **{target_name: target},
        ),
        0.1,
    )

    assert decision.source == expected_source
    assert decision.source != "line"


@pytest.mark.parametrize(
    ("target_name", "target"),
    [
        (
            "ball",
            ball_info(depth_m=0.91, distance_m=0.91),
        ),
        (
            "goal",
            goal_info(depth_m=0.51, distance_m=0.51),
        ),
        (
            "hurdle",
            hurdle_info(depth_m=1.01, distance_m=1.01),
        ),
    ],
)
def test_out_of_range_mission_does_not_preempt_corner(
    target_name,
    target,
):
    """Tracking-only detections leave the confirmed corner in control."""
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "LINE_TRACK",
        observations(
            line=confirmed_right_corner_line(),
            **{target_name: target},
        ),
        0.1,
    )

    assert decision.source == "line"
    assert decision.source_command["corner_prepare"] is True
    assert decision.source_command["corner_direction"] == "RIGHT"


def test_confirmed_side_hurdle_still_has_priority_over_close_ball():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=0.85, distance_m=0.86),
            hurdle=hurdle_info(
                bbox=[1050, 300, 1250, 500],
                offset_x_norm=0.80,
            ),
        ),
        0.1,
    )

    assert decision.source == "hurdle"
    assert decision.action == "GO"


def test_hurdle_outside_one_meter_does_not_override_close_ball():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=0.85, distance_m=0.85),
            hurdle=hurdle_info(depth_m=1.01, distance_m=1.01),
        ),
        0.1,
    )

    assert decision.source == "ball"
    assert decision.action == "STRAIGHT"


def test_hurdle_outside_one_meter_is_ignored_for_line_tracking():
    planner = MotionDecisionPlanner()
    line = line_info()
    line["filtered_heading_error_deg"] = 14.0

    for _ in range(3):
        decision = planner.plan(
            "AUTO",
            observations(
                line=line,
                hurdle=hurdle_info(
                    depth_m=1.2,
                    distance_m=1.2,
                    ground_gap_m=0.9,
                    camera_bottom_gap_m=0.5,
                    go_now=False,
                ),
            ),
            0.1,
        )

    assert decision.source == "line"
    assert decision.action == "RECOVER_RIGHT_TURN_RIGHT_1"
    assert decision.reason == "line_center_recovery"


def test_recovery_heading_deadband_is_configurable():
    planner = MotionDecisionPlanner(
        MotionDecisionConfig(recovery_heading_turn_deg=10.0)
    )
    line = line_info()
    line["filtered_heading_error_deg"] = 7.0

    decision = planner.plan(
        "LINE_TRACK",
        observations(line=line),
        0.1,
    )

    assert decision.action == "STRAIGHT"


def test_away_heading_deadband_is_forwarded_to_line_planner():
    planner = MotionDecisionPlanner(
        MotionDecisionConfig(recovery_away_heading_turn_deg=3.0)
    )
    line = line_info()
    line["filtered_lateral_offset_norm"] = -0.30
    line["filtered_heading_error_deg"] = -3.0

    decision = planner.plan(
        "LINE_TRACK",
        observations(line=line),
        0.1,
    )

    assert decision.action == "RECOVER_LEFT_TURN_LEFT_1"


def test_matching_curve_offset_override_is_forwarded_to_line_planner():
    planner = MotionDecisionPlanner(
        MotionDecisionConfig(curve_follow_max_offset_norm=0.55)
    )
    line = line_info()
    line["filtered_lateral_offset_norm"] = 0.273
    line["filtered_heading_error_deg"] = 6.3
    line["turn_angle_deg"] = 48.8
    line["turn_consistency"] = 0.9

    decisions = [
        planner.plan(
            "LINE_TRACK",
            observations(line=line),
            0.1,
        )
        for _ in range(3)
    ]

    assert decisions[-1].action == "RIGHT"


def test_hurdle_without_depth_falls_back_to_line_direction():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            hurdle=hurdle_info(depth_valid=False, depth_m=None),
        ),
        0.1,
    )

    assert decision.source == "line"
    assert decision.action == "STRAIGHT"
    assert decision.valid is True
    assert decision.reason == "line_tracking"


def test_ball_between_control_and_tracking_range_keeps_line():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=1.2, distance_m=1.2),
        ),
        0.1,
    )

    assert decision.source == "line"
    assert decision.action == "STRAIGHT"
    assert planner.ball_tracking_active is False


def test_ball_search_keeps_line_until_ball_is_inside_90cm():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "BALL_SEARCH",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=1.2, distance_m=1.2),
        ),
        0.1,
    )

    assert decision.source == "line"
    assert decision.action == "STRAIGHT"


def test_ball_beyond_1_5m_does_not_start_tracking_memory():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=1.501, distance_m=1.501),
        ),
        0.1,
    )

    assert decision.source == "line"
    assert planner.ball_tracking_active is False


def test_ball_leaving_tracking_range_clears_recovery_memory():
    planner = recovery_planner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=1.2, distance_m=1.2),
        ),
        0.1,
    )

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball={
                "detected": False,
                "note": "ball_outside_tracking_range",
            },
        ),
        0.1,
    )

    assert decision.source == "line"
    assert planner.ball_tracking_active is False


def test_missing_ball_stays_locked_and_does_not_use_line():
    planner = MotionDecisionPlanner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=0.85, distance_m=0.86),
        ),
        0.1,
    )

    decision = planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.1,
    )

    assert decision.source == "ball"
    assert decision.action == "STOP"
    assert decision.reason == "ball_not_detected"
    assert planner.ball_tracking_active is False
    assert planner.ball_recovery_centering is False


def test_90cm_takeover_uses_depth_not_hypotenuse_distance():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(
                depth_m=0.89,
                distance_m=1.70,
                horizontal_distance_m=1.60,
            ),
        ),
        0.1,
    )

    assert decision.source == "ball"
    assert decision.action == "STRAIGHT"


def test_lost_tracked_ball_stops_then_turns_toward_last_seen_side():
    planner = recovery_planner()
    visible_right = ball_info(
        depth_m=0.85,
        distance_m=0.85,
        bearing_deg=12.0,
        offset_x_norm=0.25,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), ball=visible_right),
        0.1,
    )

    stopped = planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.1,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.1,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.1,
    )
    turning = planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.1,
    )

    assert stopped.source == "ball"
    assert stopped.action == "BALL_LOST_STOP"
    assert stopped.source_command["linear_speed_mps"] == 0.0
    assert turning.action == "RECOVER_TURN_RIGHT"
    assert turning.source_command["linear_speed_mps"] == 0.0
    assert turning.source_command["angular_speed_rad_s"] > 0.0


def test_blocking_hurdle_interrupts_ball_recovery_turn():
    planner = recovery_planner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(
                depth_m=0.85,
                distance_m=0.85,
                bearing_deg=12.0,
            ),
        ),
        0.1,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.4,
    )

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball={"detected": False},
            hurdle=hurdle_info(),
        ),
        0.1,
    )

    assert decision.source == "hurdle"
    assert decision.action == "GO"


def test_reacquired_ball_inside_90cm_resumes_ball_control():
    planner = recovery_planner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=1.2, distance_m=1.2),
        ),
        0.1,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.4,
    )

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=0.88, distance_m=0.89),
        ),
        0.1,
    )

    assert decision.source == "ball"
    assert decision.action == "STRAIGHT"
    assert planner.ball_lost_elapsed_sec == 0.0


def test_reacquired_far_ball_cannot_fall_back_to_line_after_mission_entry():
    planner = recovery_planner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(
                depth_m=0.85,
                distance_m=0.85,
                bearing_deg=-15.0,
            ),
        ),
        0.1,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), ball={"detected": False}),
        0.4,
    )

    centering = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(
                depth_m=1.2,
                distance_m=1.2,
                bearing_deg=-10.0,
            ),
        ),
        0.1,
    )
    resumed = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(
                depth_m=1.2,
                distance_m=1.2,
                bearing_deg=2.0,
            ),
        ),
        0.1,
    )

    assert centering.source == "ball"
    assert centering.action == "RECOVER_TURN_LEFT"
    assert centering.source_command["linear_speed_mps"] == 0.0
    assert resumed.source == "ball"
    assert resumed.action == "STOP"
    assert resumed.reason == "ball_outside_control_range"
    assert planner.ball_recovery_centering is False


def test_goal_between_control_and_tracking_range_is_remembered():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal=goal_info(depth_m=0.8, distance_m=0.8),
        ),
        0.1,
    )

    assert decision.source == "line"
    assert planner.goal_tracking_active is True


def test_goal_leaving_tracking_range_clears_recovery_memory():
    planner = MotionDecisionPlanner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal=goal_info(depth_m=0.8, distance_m=0.8),
        ),
        0.1,
    )

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal={
                "detected": False,
                "note": "goal_outside_tracking_range",
            },
        ),
        0.1,
    )

    assert decision.source == "line"
    assert planner.goal_tracking_active is False


def test_goal_inside_50cm_takes_priority_and_approaches():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal=goal_info(depth_m=0.49, distance_m=0.49),
        ),
        0.1,
    )

    assert decision.source == "goal"
    assert decision.action == "STRAIGHT_3"


def test_goal_search_keeps_line_until_goal_is_inside_50cm():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "GOAL_SEARCH",
        observations(
            line=line_info(),
            goal=goal_info(depth_m=1.0, distance_m=1.0),
        ),
        0.1,
    )

    assert decision.source == "line"


def test_lost_goal_stops_then_turns_toward_last_seen_side():
    planner = MotionDecisionPlanner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal=goal_info(
                depth_m=0.49,
                distance_m=0.49,
                bearing_deg=-12.0,
                offset_x_norm=-0.25,
            ),
        ),
        0.1,
    )

    stopped = planner.plan(
        "AUTO",
        observations(line=line_info(), goal={"detected": False}),
        0.1,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), goal={"detected": False}),
        0.3,
    )
    turning = planner.plan(
        "AUTO",
        observations(line=line_info(), goal={"detected": False}),
        0.1,
    )

    assert stopped.source == "goal"
    assert stopped.action == "GOAL_LOST_STOP"
    assert stopped.source_command["linear_speed_mps"] == 0.0
    assert turning.action == "RECOVER_GOAL_TURN_LEFT"
    assert turning.source_command["angular_speed_rad_s"] < 0.0


def test_reacquired_far_goal_cannot_fall_back_to_line_after_mission_entry():
    planner = MotionDecisionPlanner()
    planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal=goal_info(
                depth_m=0.49,
                distance_m=0.49,
                bearing_deg=15.0,
            ),
        ),
        0.1,
    )
    planner.plan(
        "AUTO",
        observations(line=line_info(), goal={"detected": False}),
        0.4,
    )

    centering = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal=goal_info(
                depth_m=1.0,
                distance_m=1.0,
                bearing_deg=10.0,
            ),
        ),
        0.1,
    )
    resumed = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            goal=goal_info(
                depth_m=1.0,
                distance_m=1.0,
                bearing_deg=2.0,
            ),
        ),
        0.1,
    )

    assert centering.source == "goal"
    assert centering.action == "RECOVER_GOAL_TURN_RIGHT"
    assert resumed.source == "goal"
    assert resumed.action == "WAIT"
    assert resumed.reason == "goal_outside_control_range"
    assert planner.goal_recovery_centering is False


def test_hurdle_go_is_normalized_as_acknowledged_sdk_event():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "HURDLE_APPROACH",
        observations(hurdle=hurdle_info()),
        0.1,
    )

    assert decision.source == "hurdle"
    assert decision.action == "GO"
    assert decision.sdk_motion_requested is True
    assert decision.requires_ack is True


def test_hurdle_uses_line_intersection_without_entering_line_mode():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "AUTO",
        observations(
            line=hurdle_route_line(x=820),
            hurdle=hurdle_info(
                ground_gap_m=0.45,
                camera_bottom_gap_m=0.20,
                go_now=False,
            ),
        ),
        0.1,
    )

    assert decision.source == "hurdle"
    assert decision.action == "TURN_RIGHT"
    assert decision.reason == "align_to_hurdle_line_intersection"
    assert decision.source_command["path_reference_valid"] is True
    assert decision.source_command["line_mode_allowed"] is False
    assert "line_guidance" not in decision.source_command


def test_hurdle_path_reference_is_held_for_short_line_dropout():
    planner = MotionDecisionPlanner()
    hurdle = hurdle_info(
        ground_gap_m=0.45,
        camera_bottom_gap_m=0.20,
        go_now=False,
    )
    planner.plan(
        "AUTO",
        observations(line=hurdle_route_line(x=820), hurdle=hurdle),
        0.1,
    )

    held = planner.plan(
        "AUTO",
        observations(line={"detected": False}, hurdle=hurdle),
        0.2,
    )

    assert held.source == "hurdle"
    assert held.action == "TURN_RIGHT"
    assert held.source_command["path_reference_source"] == "held"


def test_hurdle_lock_never_falls_back_to_line_before_jump():
    planner = MotionDecisionPlanner()
    planner.plan(
        "AUTO",
        observations(
            line=hurdle_route_line(),
            hurdle=hurdle_info(
                ground_gap_m=0.45,
                camera_bottom_gap_m=0.20,
                go_now=False,
            ),
        ),
        0.1,
    )

    lost = planner.plan(
        "AUTO",
        observations(line=line_info(), hurdle={"detected": False}),
        1.0,
    )

    assert lost.source == "hurdle"
    assert lost.action == "WAIT"
    assert lost.reason == "hurdle_not_detected"


def test_hurdle_lock_releases_after_go_and_explicit_phase_change():
    planner = MotionDecisionPlanner()
    go = planner.plan(
        "HURDLE_APPROACH",
        observations(hurdle=hurdle_info()),
        0.1,
    )
    assert go.action == "GO"

    resumed = planner.plan(
        "LINE_TRACK",
        observations(line=line_info(), hurdle=hurdle_info()),
        0.1,
    )

    assert resumed.source == "line"
    assert resumed.action == "STRAIGHT"


def test_ball_lock_releases_only_after_pickup_and_phase_change():
    planner = MotionDecisionPlanner()
    pickup = planner.plan(
        "AUTO",
        observations(
            line=line_info(),
            ball=ball_info(
                depth_m=0.07,
                distance_m=0.07,
                pickup_ready=True,
                pickup_now=True,
            ),
        ),
        0.1,
    )
    assert pickup.action == "PICKUP_NOW"

    resumed = planner.plan(
        "LINE_TRACK",
        observations(
            line=line_info(),
            ball=ball_info(depth_m=0.07, distance_m=0.07),
        ),
        0.1,
    )

    assert resumed.source == "line"
    assert resumed.action == "STRAIGHT"


def test_goal_lock_releases_only_after_shot_and_phase_change():
    planner = MotionDecisionPlanner()
    shot = planner.plan(
        "AUTO",
        observations(line=line_info(), goal=goal_info()),
        0.1,
    )
    assert shot.action == "SHOT"

    resumed = planner.plan(
        "LINE_TRACK",
        observations(line=line_info(), goal=goal_info()),
        0.1,
    )

    assert resumed.source == "line"
    assert resumed.action == "STRAIGHT"


def test_right_curve_remains_available_after_goal_is_scored():
    """Post-goal line phases can still execute the required right curve."""
    planner = MotionDecisionPlanner()
    shot = planner.plan(
        "GOAL_APPROACH",
        observations(goal=goal_info()),
        0.1,
    )
    assert shot.action == "SHOT"

    right_curve = line_info(
        filtered_heading_error_deg=14.0,
        turn_angle_deg=30.0,
        turn_consistency=0.95,
    )
    decisions = [
        planner.plan(
            "LINE_TRACK",
            observations(line=right_curve, goal=goal_info()),
            0.1,
        )
        for _ in range(3)
    ]

    assert decisions[-1].source == "line"
    assert decisions[-1].action == "RIGHT"


def test_line_phase_reuses_existing_line_planner():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "LINE_TRACK",
        observations(line=line_info()),
        0.1,
    )

    assert decision.source == "line"
    assert decision.action == "STRAIGHT"
    assert decision.valid is True


@pytest.mark.parametrize(
    "phase",
    ["follow_line_to_ball_a", "FOLLOW_LINE_TO_GOAL_A"],
)
def test_mission_estimator_line_phases_resume_after_reacquisition(phase):
    planner = MotionDecisionPlanner()

    lost = planner.plan(phase, observations(line={"detected": False}), 0.1)
    reacquired = planner.plan(phase, observations(line=line_info()), 0.1)

    assert lost.source == "line"
    assert lost.action == "STOP"
    assert reacquired.source == "line"
    assert reacquired.action == "STRAIGHT"
    assert reacquired.valid is True


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("pick_ball_a", "ball"), ("score_goal_a", "goal")],
)
def test_mission_estimator_object_phases_are_recognized(phase, expected):
    assert MotionDecisionPlanner.source_for_phase(phase) == expected


def test_unknown_phase_fails_safe():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "NOT_A_PHASE",
        observations(ball=ball_info()),
        0.1,
    )

    assert decision.source == "none"
    assert decision.action == "WAIT"
    assert decision.valid is False


def test_search_phase_tracks_line_until_target_appears():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "BALL_SEARCH",
        observations(line=line_info()),
        0.1,
    )

    assert decision.source == "line"
    assert decision.action == "STRAIGHT"


def test_lock_phase_waits_for_cpp_motion_status():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "HURDLE_LOCK",
        observations(line=line_info(), hurdle=hurdle_info()),
        0.1,
    )

    assert decision.source == "none"
    assert decision.action == "WAIT"
    assert decision.reason == "mission_locked_waiting_for_motion_status"


def test_line_lock_keeps_publishing_continuous_line_guidance():
    planner = MotionDecisionPlanner()

    decision = planner.plan(
        "LINE_LOCK",
        observations(line=line_info()),
        0.1,
    )

    assert decision.source == "line"
    assert decision.action == "STRAIGHT"
    assert decision.valid is True
    assert decision.requires_ack is False
