from step.visual_motion_advisor import numbered_turn_angle
from step.visual_motion_advisor import numbered_turn_for_angle
from step.visual_motion_advisor import VisualMotionAdvisor


def test_numbered_turn_uses_deployed_motion_suffixes():
    assert numbered_turn_for_angle(-12.0) == "TURN_LEFT_2"
    assert numbered_turn_for_angle(33.0) == "TURN_RIGHT_6"
    assert numbered_turn_for_angle(4.0) is None
    assert numbered_turn_angle("TURN_LEFT_10") == -75.0
    assert numbered_turn_angle("TURN_RIGHT_15") == 90.0


def test_ball_turn_does_not_require_depth():
    advisor = VisualMotionAdvisor()
    suggestion = advisor.suggest_ball(
        {
            "detected": True,
            "steering_angle_deg": 31.0,
            "depth_valid": False,
        },
        now=1.0,
        observation_time=1.0,
    )
    assert suggestion is not None
    assert suggestion.action == "TURN_RIGHT_6"


def test_ball_forward_suggestion_requires_depth():
    advisor = VisualMotionAdvisor()
    no_depth = advisor.suggest_ball(
        {
            "detected": True,
            "steering_angle_deg": 0.0,
            "depth_valid": False,
        },
        now=1.0,
        observation_time=1.0,
    )
    with_depth = advisor.suggest_ball(
        {
            "detected": True,
            "steering_angle_deg": 0.0,
            "depth_valid": True,
            "depth_m": 0.50,
        },
        now=2.0,
        observation_time=2.0,
    )
    assert no_depth is not None
    assert no_depth.action == "HOLD_NO_DEPTH"
    assert with_depth is not None
    assert with_depth.action == "STRAIGHT_3"


def test_ball_loss_is_only_a_visual_recovery_suggestion():
    advisor = VisualMotionAdvisor()
    advisor.suggest_ball(
        {
            "detected": True,
            "steering_angle_deg": -18.0,
            "depth_valid": True,
            "depth_m": 0.60,
        },
        now=1.0,
        observation_time=1.0,
    )
    hold = advisor.suggest_ball(None, now=1.2, observation_time=None)
    search = advisor.suggest_ball(None, now=1.6, observation_time=None)
    assert hold is not None
    assert hold.action == "BALL_LOST_HOLD"
    assert search is not None
    assert search.action == "FIND_BALL_LEFT"


def test_goal_and_hurdle_suggestions_use_analyzer_facts():
    advisor = VisualMotionAdvisor()
    goal = advisor.suggest_goal(
        {
            "detected": True,
            "offset_x_norm": 0.2,
            "depth_valid": False,
        }
    )
    hurdle = advisor.suggest_hurdle(
        {
            "detected": True,
            "depth_valid": True,
            "ground_distance_m": 0.5,
            "state": "APPROACH",
        }
    )
    assert goal is not None
    assert goal.action == "TURN_RIGHT"
    assert hurdle is not None
    assert hurdle.action == "STRAIGHT_3"
