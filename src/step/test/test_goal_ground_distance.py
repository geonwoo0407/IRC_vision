"""Tests for goal ground-plane distance reporting."""

import pytest

from step.goal_analyzer import GoalAnalyzer


def test_goal_projection_reports_ground_plane_distance():
    """Expose pitch-corrected robot-floor range separately from optical Z."""
    analyzer = object.__new__(GoalAnalyzer)
    analyzer.fx = 600.0
    analyzer.fy = 600.0
    analyzer.cx = 640.0
    analyzer.cy = 360.0
    analyzer.camera_pitch_down_deg = 45.0
    analyzer.camera_forward_offset_m = 0.0

    projection = analyzer._project(640, 360, 1.0)

    assert projection[3] == pytest.approx(1.0)
    assert projection[4] == pytest.approx(2 ** -0.5)


def test_rgb_goal_remains_candidate_without_depth():
    """Keep goal geometry available while aligned depth is missing."""
    analyzer = object.__new__(GoalAnalyzer)
    analyzer.min_confidence = 0.55
    analyzer.direction_deadband_norm = 0.04
    analyzer.max_valid_depth_m = 6.0
    analyzer.score_center_tolerance_norm = 0.10
    analyzer.score_target_depth_m = 0.25
    analyzer.score_depth_tolerance_m = 0.05
    analyzer.approach_depth_m = 0.5
    analyzer.fx = None
    analyzer.fy = None
    analyzer.cx = None
    analyzer.cy = None
    analyzer._sample_goal_depth_m = lambda _bbox: (None, False, 0)
    detection = {
        "confidence": 0.83,
        "bbox": [300, 140, 980, 600],
        "center": [640, 370],
    }

    candidate = analyzer._build_candidate(detection, 1280, 720)

    assert candidate is not None
    assert candidate.depth_valid is False
    assert candidate.depth_m is None
    state = analyzer._goal_state(candidate)
    assert state[0] == "NO_DEPTH"
