"""
Compute display-only motion suggestions from vision observations.

This module never publishes a ROS command and never requests a robot motion.
It exists only so the detector UI can explain what the current geometry would
suggest while the Algorithm package remains the sole motion owner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .approach_distance import approach_motion_for_distance


TURN_STEP_DEG = 15.0
TURN_MAX_LEVEL = 6
TURN_ACTION_SUFFIXES = {
    "LEFT": (2, 4, 6, 8, 10, 13),
    "RIGHT": (4, 6, 8, 10, 12, 15),
}


@dataclass(frozen=True)
class VisualSuggestion:
    """One non-command recommendation for the detector overlay."""

    action: str
    reason: str


def _number(data: dict[str, object] | None, key: str) -> float | None:
    if data is None:
        return None
    value = data.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numbered_turn_for_angle(
    angle_deg: float,
    deadband_deg: float = 5.0,
) -> str | None:
    """Return a display-only numbered turn using deployed angle labels."""
    if abs(angle_deg) <= deadband_deg:
        return None
    level = math.floor((abs(angle_deg) + TURN_STEP_DEG / 2.0) / TURN_STEP_DEG)
    level = max(1, min(TURN_MAX_LEVEL, level))
    direction = "RIGHT" if angle_deg > 0.0 else "LEFT"
    suffix = TURN_ACTION_SUFFIXES[direction][level - 1]
    return f"TURN_{direction}_{suffix}"


def numbered_turn_angle(action: str) -> float | None:
    """Decode a display turn label into its signed nominal angle."""
    normalized = action.strip().upper()
    for direction, sign in (("LEFT", -1.0), ("RIGHT", 1.0)):
        prefix = f"TURN_{direction}_"
        if not normalized.startswith(prefix):
            continue
        try:
            suffix = int(normalized.removeprefix(prefix))
        except ValueError:
            return None
        suffixes = TURN_ACTION_SUFFIXES[direction]
        if suffix not in suffixes:
            return None
        return sign * (suffixes.index(suffix) + 1) * TURN_STEP_DEG
    return None


class VisualMotionAdvisor:
    """Suggest overlay text without changing navigation or mission state."""

    def __init__(
        self,
        *,
        ball_control_depth_m: float = 1.5,
        ball_lost_hold_sec: float = 0.35,
        ball_search_timeout_sec: float = 8.0,
    ) -> None:
        self.ball_control_depth_m = max(0.0, ball_control_depth_m)
        self.ball_lost_hold_sec = max(0.0, ball_lost_hold_sec)
        self.ball_search_timeout_sec = max(
            self.ball_lost_hold_sec,
            ball_search_timeout_sec,
        )
        self.last_ball_seen_time: float | None = None
        self.last_ball_observation_time: float | None = None
        self.last_ball_side = "RIGHT"
        self.last_ball_depth_m: float | None = None

    def suggest_ball(
        self,
        info: dict[str, object] | None,
        *,
        now: float,
        observation_time: float | None,
    ) -> VisualSuggestion | None:
        """Suggest ball alignment, approach, pickup, or visual search text."""
        detected = bool(info and info.get("detected", False))
        if detected and info is not None:
            self._remember_ball(info, now, observation_time)
            if bool(info.get("pickup_now", False)):
                return VisualSuggestion("PICKUP_NOW", "pickup_window_ready")

            steering = _number(info, "steering_angle_deg")
            if steering is None:
                steering = _number(info, "bearing_deg")
            if steering is None:
                offset = _number(info, "offset_x_norm")
                steering = offset * 35.0 if offset is not None else None
            if steering is not None:
                turn = numbered_turn_for_angle(steering)
                if turn is not None:
                    return VisualSuggestion(turn, "align_ball_path")

            depth = _number(info, "depth_m")
            depth_valid = bool(info.get("depth_valid", False))
            if not depth_valid or depth is None:
                return VisualSuggestion(
                    "HOLD_NO_DEPTH",
                    "ball_centered_without_valid_depth",
                )
            if depth > self.ball_control_depth_m:
                return VisualSuggestion(
                    "TRACK_ONLY",
                    "ball_outside_visual_control_range",
                )
            return VisualSuggestion(
                approach_motion_for_distance(depth),
                "ball_aligned_visual_approach",
            )

        if info and bool(info.get("raw_detected", False)):
            return VisualSuggestion(
                "WAIT_CONFIRMATION",
                "ball_confirmation_pending",
            )
        return self._lost_ball_suggestion(now)

    def suggest_goal(
        self,
        info: dict[str, object] | None,
    ) -> VisualSuggestion | None:
        """Suggest display text from confirmed goal geometry."""
        if not info or not bool(info.get("detected", False)):
            if info and bool(info.get("raw_detected", False)):
                return VisualSuggestion(
                    "WAIT_CONFIRMATION",
                    "goal_confirmation_pending",
                )
            return None
        offset = _number(info, "offset_x_norm")
        if offset is not None and abs(offset) > 0.10:
            action = "TURN_RIGHT" if offset > 0.0 else "TURN_LEFT"
            return VisualSuggestion(action, "align_goal_center")
        depth = _number(info, "depth_m")
        if not bool(info.get("depth_valid", False)) or depth is None:
            return VisualSuggestion(
                "HOLD_NO_DEPTH",
                "goal_centered_without_valid_depth",
            )
        if bool(info.get("score_now", False)):
            return VisualSuggestion("SHOT", "goal_score_window_ready")
        if depth > 0.30:
            return VisualSuggestion(
                approach_motion_for_distance(depth),
                "goal_aligned_visual_approach",
            )
        if depth < 0.20:
            return VisualSuggestion("RETREAT_GOAL", "goal_too_close")
        return VisualSuggestion(
            "WAIT_SCORE_CONFIRMATION",
            "goal_score_confirmation_pending",
        )

    def suggest_hurdle(
        self,
        info: dict[str, object] | None,
        *,
        path_offset_x_norm: float | None = None,
    ) -> VisualSuggestion | None:
        """Suggest display text from confirmed hurdle geometry."""
        if not info or not bool(info.get("detected", False)):
            if info and bool(info.get("raw_detected", False)):
                return VisualSuggestion(
                    "WAIT_CONFIRMATION",
                    "hurdle_confirmation_pending",
                )
            return None
        if (
            path_offset_x_norm is not None
            and abs(path_offset_x_norm) > 0.10
        ):
            action = (
                "TURN_RIGHT"
                if path_offset_x_norm > 0.0
                else "TURN_LEFT"
            )
            return VisualSuggestion(action, "align_hurdle_path")
        angle = _number(info, "hurdle_angle_deg")
        if angle is not None and abs(angle) > 8.0:
            action = "ALIGN_LEFT" if angle > 0.0 else "ALIGN_RIGHT"
            return VisualSuggestion(action, "align_hurdle_parallel")
        ground_distance = _number(info, "ground_distance_m")
        if not bool(info.get("depth_valid", False)) or ground_distance is None:
            return VisualSuggestion(
                "HOLD_NO_DEPTH",
                "hurdle_without_valid_ground_distance",
            )
        if bool(info.get("go_now", False)):
            return VisualSuggestion("GO", "hurdle_go_window_ready")
        state = str(info.get("state", "")).upper()
        if state == "APPROACH":
            return VisualSuggestion(
                approach_motion_for_distance(ground_distance),
                "hurdle_aligned_visual_approach",
            )
        return VisualSuggestion(
            "WAIT_GO_CONFIRMATION",
            "hurdle_go_confirmation_pending",
        )

    def ball_recovery_active(self, now: float) -> bool:
        """Return whether a recent confirmed ball still has display memory."""
        if self.last_ball_seen_time is None:
            return False
        return now - self.last_ball_seen_time <= self.ball_search_timeout_sec

    def _remember_ball(
        self,
        info: dict[str, object],
        now: float,
        observation_time: float | None,
    ) -> None:
        sample_time = observation_time if observation_time is not None else now
        if self.last_ball_observation_time == sample_time:
            return
        self.last_ball_observation_time = sample_time
        self.last_ball_seen_time = sample_time
        steering = _number(info, "steering_angle_deg")
        if steering is None:
            steering = _number(info, "bearing_deg")
        if steering is not None and abs(steering) > 1.0:
            self.last_ball_side = "RIGHT" if steering > 0.0 else "LEFT"
        depth = _number(info, "depth_m")
        if bool(info.get("depth_valid", False)) and depth is not None:
            self.last_ball_depth_m = depth

    def _lost_ball_suggestion(self, now: float) -> VisualSuggestion | None:
        if self.last_ball_seen_time is None:
            return None
        elapsed = max(0.0, now - self.last_ball_seen_time)
        if elapsed > self.ball_search_timeout_sec:
            return VisualSuggestion(
                "BALL_SEARCH_TIMEOUT",
                "ball_visual_search_timed_out",
            )
        if elapsed <= self.ball_lost_hold_sec:
            return VisualSuggestion("BALL_LOST_HOLD", "brief_ball_loss_hold")
        initial_turn_end = self.ball_lost_hold_sec + 1.20
        if elapsed <= initial_turn_end:
            return VisualSuggestion(
                f"FIND_BALL_{self.last_ball_side}",
                "search_last_seen_ball_side",
            )
        forward_end = initial_turn_end + 0.80
        if (
            self.last_ball_depth_m is not None
            and self.last_ball_depth_m > 0.25
            and elapsed <= forward_end
        ):
            return VisualSuggestion(
                "FIND_BALL_FORWARD",
                "search_last_seen_ball_position",
            )
        sweep_index = int(max(0.0, elapsed - forward_end) / 1.20)
        direction = self.last_ball_side
        if sweep_index % 2 == 1:
            direction = "LEFT" if direction == "RIGHT" else "RIGHT"
        return VisualSuggestion(
            f"FIND_BALL_{direction}",
            "visual_ball_search_sweep",
        )
