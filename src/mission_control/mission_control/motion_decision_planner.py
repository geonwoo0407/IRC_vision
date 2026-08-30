#!/usr/bin/env python3
"""Select one mission command from the existing navigation planners."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from step.ball_navigation_planner import BallNavigationConfig
from step.ball_navigation_planner import BallNavigationPlanner
from step.goal_navigation_planner import GoalNavigationPlanner
from step.hurdle_navigation_planner import HurdleNavigationPlanner
from step.line_navigation_planner import LineNavigationPlanner
from step.line_navigation_planner import NavigationConfig

from .hurdle_line_fusion import build_hurdle_path_reference


@dataclass(frozen=True)
class MotionDecision:
    """One normalized command selected from a mission-specific planner."""

    phase: str
    source: str
    action: str
    valid: bool
    reason: str
    sdk_motion_requested: bool
    requires_ack: bool
    source_command: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "phase": self.phase,
            "source": self.source,
            "action": self.action,
            "valid": self.valid,
            "reason": self.reason,
            "sdk_motion_requested": self.sdk_motion_requested,
            "requires_ack": self.requires_ack,
            "source_command": self.source_command,
        }


@dataclass(frozen=True)
class MotionDecisionConfig:
    """Tunable mission-selection and lost-ball recovery limits."""

    enable_ball_lost_recovery: bool = True
    recovery_heading_turn_deg: float = 10.0
    recovery_away_heading_turn_deg: float = 3.0
    curve_follow_max_offset_norm: float = 0.55
    ball_tracking_range_m: float = 1.5
    ball_control_range_m: float = 1.5
    ball_lost_stop_sec: float = 0.35
    ball_recovery_timeout_sec: float = 8.0
    ball_recovery_turn_rad_s: float = 0.22
    ball_recovery_command_sec: float = 0.40
    ball_recovery_initial_turn_max_sec: float = 2.0
    ball_recovery_forward_sec: float = 0.40
    ball_recovery_forward_min_depth_m: float = 0.35
    ball_recovery_forward_max_bearing_deg: float = 25.0
    ball_recovery_sweep_sec: float = 1.20
    ball_recovery_direction_deadband_deg: float = 1.0
    ball_reacquire_center_deg: float = 5.0
    goal_tracking_range_m: float = 1.0
    goal_control_range_m: float = 0.5
    hurdle_control_range_m: float = 1.0
    hurdle_path_reference_hold_sec: float = 0.50
    goal_lost_stop_sec: float = 0.35
    goal_recovery_timeout_sec: float = 8.0
    goal_recovery_turn_rad_s: float = 0.22
    goal_recovery_command_sec: float = 0.40
    goal_reacquire_center_deg: float = 5.0
    goal_reacquire_center_norm: float = 0.10


class MotionDecisionPlanner:
    """Run one existing planner according to the active mission phase."""

    AUTO_PRIORITY = ("line",)
    TERMINAL_ACTIONS = {
        ("ball", "PICKUP_NOW"),
        ("goal", "SHOT"),
        ("hurdle", "GO"),
    }

    def __init__(
        self,
        config: MotionDecisionConfig | None = None,
    ) -> None:
        self.config = config or MotionDecisionConfig()
        self.line_planner = LineNavigationPlanner(
            NavigationConfig(
                recovery_heading_turn_deg=(
                    self.config.recovery_heading_turn_deg
                ),
                recovery_away_heading_turn_deg=(
                    self.config.recovery_away_heading_turn_deg
                ),
                curve_follow_max_offset_norm=(
                    self.config.curve_follow_max_offset_norm
                ),
            )
        )
        self.ball_planner = BallNavigationPlanner(
            BallNavigationConfig(
                control_start_depth_m=self.config.ball_control_range_m,
            )
        )
        self.goal_planner = GoalNavigationPlanner()
        self.hurdle_planner = HurdleNavigationPlanner()
        self.previous_source = "none"
        self.ball_tracking_active = False
        self.ball_recovery_centering = False
        self.ball_lost_elapsed_sec = 0.0
        self.last_ball_bearing_deg: float | None = None
        self.last_ball_offset_x_norm: float | None = None
        self.last_ball_depth_m: float | None = None
        self.last_ball_turn_direction = "RIGHT"
        self.ball_recovery_timed_out = False
        self.ball_lock_active = False
        self.ball_terminal_requested = False
        self.ball_ignore_until_clear = False
        self.goal_tracking_active = False
        self.goal_recovery_centering = False
        self.goal_lost_elapsed_sec = 0.0
        self.last_goal_bearing_deg: float | None = None
        self.last_goal_offset_x_norm: float | None = None
        self.last_goal_turn_direction = "RIGHT"
        self.goal_lock_active = False
        self.goal_terminal_requested = False
        self.goal_ignore_until_clear = False
        self.hurdle_lock_active = False
        self.hurdle_go_requested = False
        self.hurdle_ignore_until_clear = False
        self.last_hurdle_path_reference: dict[str, Any] | None = None
        self.hurdle_path_reference_age_sec = 0.0

    @staticmethod
    def source_for_phase(phase: str) -> str | None:
        """Map a mission phase name to the sensor that owns that phase."""
        normalized = phase.strip().upper()
        if normalized == "AUTO":
            return None
        # Accept both the short controller phases (LINE_TRACK, BALL_SEARCH,
        # ...) and the semantic names published by mission_state_estimator
        # (follow_line_to_ball_a, pick_ball_a, score_goal_a).  Previously the
        # latter were classified as unknown, so the decision stayed at WAIT
        # even after the line had been reacquired.
        if normalized.startswith("BALL") or normalized.startswith("PICK"):
            return "ball"
        if (
            normalized.startswith("GOAL")
            or normalized.startswith("SHOOT")
            or normalized.startswith("SCORE_GOAL")
        ):
            return "goal"
        if normalized.startswith("HURDLE") or normalized.startswith("JUMP"):
            return "hurdle"
        if (
            normalized.startswith("LINE")
            or normalized.startswith("FOLLOW_LINE")
            or normalized == "FINISH"
        ):
            return "line"
        return "none"

    def plan(
        self,
        phase: str,
        observations: dict[str, dict[str, Any] | None],
        dt_sec: float,
    ) -> MotionDecision:
        """Select a source and normalize its planner-specific command."""
        normalized_phase = phase.strip().upper() or "AUTO"
        self._update_ball_tracking(observations.get("ball"), dt_sec)
        self._update_goal_tracking(observations.get("goal"), dt_sec)
        self._update_hurdle_lock(
            normalized_phase,
            observations.get("hurdle"),
        )
        self._update_object_locks(normalized_phase, observations)
        # Continuous line guidance does not require an SDK completion ACK.
        # Some external FSMs still publish LINE_LOCK after receiving a walking
        # command; treating that like a terminal action leaves the robot stuck
        # at WAIT forever.  Keep locks only for discrete object motions.
        if (
            normalized_phase.endswith("_LOCK")
            and self.source_for_phase(normalized_phase) != "line"
        ):
            self._reset_previous_source()
            return MotionDecision(
                phase=normalized_phase,
                source="none",
                action="WAIT",
                valid=False,
                reason="mission_locked_waiting_for_motion_status",
                sdk_motion_requested=False,
                requires_ack=False,
                source_command={},
            )

        source = self._select_source(normalized_phase, observations)
        if source == "none":
            self._reset_previous_source()
            return MotionDecision(
                phase=normalized_phase,
                source="none",
                action="WAIT",
                valid=False,
                reason="no_fresh_detected_target",
                sdk_motion_requested=False,
                requires_ack=False,
                source_command={},
            )

        if source != self.previous_source:
            self._reset_source(source)
        self.previous_source = source
        info = observations.get(source)
        hurdle_reference: dict[str, Any] | None = None
        if source == "hurdle":
            info, hurdle_reference = self._hurdle_observation_with_path(
                info,
                observations.get("line"),
                dt_sec,
            )
        command = self._plan_source(source, info, dt_sec)
        if hurdle_reference is not None:
            command.update(hurdle_reference)
        action_key = "motion" if source in {"line", "ball"} else "action"
        action = str(command.get(action_key, "WAIT"))
        terminal = (source, action) in self.TERMINAL_ACTIONS
        requested = terminal and bool(
            command.get("sdk_motion_requested", terminal)
        )
        if source == "hurdle" and action == "GO":
            self.hurdle_go_requested = True
        elif source == "ball" and action == "PICKUP_NOW":
            self.ball_terminal_requested = True
        elif source == "goal" and action == "SHOT":
            self.goal_terminal_requested = True
        return MotionDecision(
            phase=normalized_phase,
            source=source,
            action=action,
            valid=bool(command.get("valid", False)),
            reason=str(command.get("reason", "unknown")),
            sdk_motion_requested=requested,
            requires_ack=terminal,
            source_command=command,
        )

    def _hurdle_observation_with_path(
        self,
        hurdle_info: dict[str, Any] | None,
        line_info: dict[str, Any] | None,
        dt_sec: float,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Attach hurdle-owned path geometry without running line planner."""
        reference = build_hurdle_path_reference(hurdle_info, line_info)
        if bool(reference.get("path_reference_valid", False)):
            self.last_hurdle_path_reference = dict(reference)
            self.hurdle_path_reference_age_sec = 0.0
        else:
            self.hurdle_path_reference_age_sec += max(0.0, dt_sec)
            if (
                hurdle_info is not None
                and bool(hurdle_info.get("detected", False))
                and self.last_hurdle_path_reference is not None
                and self.hurdle_path_reference_age_sec
                <= self.config.hurdle_path_reference_hold_sec
            ):
                reference = dict(self.last_hurdle_path_reference)
                reference["path_reference_source"] = "held"
                reference["path_reference_reason"] = (
                    "temporarily_held_line_hurdle_intersection"
                )
                reference["path_reference_age_sec"] = round(
                    self.hurdle_path_reference_age_sec,
                    3,
                )

        if hurdle_info is None:
            return None, reference
        enriched = dict(hurdle_info)
        enriched.update(reference)
        return enriched, reference

    def _update_hurdle_lock(
        self,
        phase: str,
        hurdle_info: dict[str, Any] | None,
    ) -> None:
        """Latch hurdle ownership until an acknowledged jump changes phase."""
        requested = self.source_for_phase(phase)
        post_jump_phase_change = bool(
            self.hurdle_lock_active
            and self.hurdle_go_requested
            and requested in {"line", "ball", "goal"}
            and not phase.endswith("_LOCK")
        )
        if phase in {"HURDLE_DONE", "JUMP_DONE", "POST_HURDLE"}:
            post_jump_phase_change = True
        if post_jump_phase_change:
            self.hurdle_lock_active = False
            self.hurdle_go_requested = False
            self.hurdle_ignore_until_clear = True
            self.last_hurdle_path_reference = None
            self.hurdle_path_reference_age_sec = 0.0
            return
        if self.hurdle_ignore_until_clear:
            if not self._confirmed_hurdle(hurdle_info):
                self.hurdle_ignore_until_clear = False
            return
        if self._confirmed_hurdle(hurdle_info):
            self.hurdle_lock_active = True

    def _update_object_locks(
        self,
        phase: str,
        observations: dict[str, dict[str, Any] | None],
    ) -> None:
        """Keep ball/goal ownership after control-range mission entry."""
        requested = self.source_for_phase(phase)
        explicit_next_source = (
            requested
            if requested in {"line", "ball", "goal", "hurdle"}
            and not phase.endswith("_LOCK")
            else None
        )

        if (
            self.ball_lock_active
            and self.ball_terminal_requested
            and explicit_next_source not in {None, "ball"}
        ):
            self.ball_lock_active = False
            self.ball_terminal_requested = False
            self.ball_ignore_until_clear = True
            self._clear_ball_tracking()
        if (
            self.goal_lock_active
            and self.goal_terminal_requested
            and explicit_next_source not in {None, "goal"}
        ):
            self.goal_lock_active = False
            self.goal_terminal_requested = False
            self.goal_ignore_until_clear = True

        ball_info = observations.get("ball")
        goal_info = observations.get("goal")
        if self.ball_ignore_until_clear:
            if not self._ball_is_inside_control_range(ball_info):
                self.ball_ignore_until_clear = False
        if self.goal_ignore_until_clear:
            if not self._goal_is_inside_control_range(goal_info):
                self.goal_ignore_until_clear = False

        if self.ball_lock_active or self.goal_lock_active:
            return
        if self.hurdle_lock_active:
            return
        # A line phase only says what to do while no mission target is close
        # enough to control.  It must not pin ownership to line/corner motion:
        # a ball or goal entering its control range immediately preempts both
        # ordinary line following and a confirmed corner approach.  Explicit
        # object phases still remain exclusive so, for example, a visible ball
        # cannot steal ownership during GOAL_APPROACH.
        phase_allows_ball = requested in {None, "line", "ball"}
        phase_allows_goal = requested in {None, "line", "goal"}
        if (
            phase_allows_ball
            and not self.ball_ignore_until_clear
            and self._ball_is_inside_control_range(ball_info)
        ):
            self.ball_lock_active = True
        elif (
            phase_allows_goal
            and not self.goal_ignore_until_clear
            and self._goal_is_inside_control_range(goal_info)
        ):
            self.goal_lock_active = True

    def _select_source(
        self,
        phase: str,
        observations: dict[str, dict[str, Any] | None],
    ) -> str:
        if self.hurdle_lock_active:
            return "hurdle"
        if self.ball_lock_active:
            return "ball"
        if self.goal_lock_active:
            return "goal"
        requested = self.source_for_phase(phase)
        if requested is None:
            return self._select_auto_source(observations)
        if requested == "none":
            return "none"
        if phase.endswith("_SEARCH"):
            target = observations.get(requested)
            if requested == "ball":
                if self._ball_is_inside_control_range(target):
                    return "ball"
            elif requested == "goal":
                if self._goal_is_inside_control_range(target):
                    return "goal"
            elif target is not None and bool(target.get("detected", False)):
                return requested
            line = observations.get("line")
            if line is not None and bool(line.get("detected", False)):
                return "line"
            return "none"
        return requested

    def _select_auto_source(
        self,
        observations: dict[str, dict[str, Any] | None],
    ) -> str:
        if self.hurdle_lock_active:
            return "hurdle"
        if self.ball_lock_active:
            return "ball"
        if self.goal_lock_active:
            return "goal"
        ball = observations.get("ball")
        if self._ball_is_inside_control_range(ball):
            return "ball"
        goal = observations.get("goal")
        if self._goal_is_inside_control_range(goal):
            return "goal"
        for source in self.AUTO_PRIORITY:
            info = observations.get(source)
            if info is not None and bool(info.get("detected", False)):
                return source
        return "none"

    def _confirmed_hurdle(self, info: dict[str, Any] | None) -> bool:
        """Return true for a confirmed hurdle inside its control range."""
        if info is None or not bool(info.get("detected", False)):
            return False
        if (
            "confirmation_confirmed" in info
            and not bool(info.get("confirmation_confirmed", False))
        ):
            return False
        depth = self._number(info, "depth_m")
        return bool(
            info.get("depth_valid", False)
            and depth is not None
            and depth <= self.config.hurdle_control_range_m
        )

    def _plan_source(
        self,
        source: str,
        info: dict[str, Any] | None,
        dt_sec: float,
    ) -> dict[str, Any]:
        if source == "line":
            command = (
                self.line_planner.stop("waiting_for_line_info")
                if info is None
                else self.line_planner.plan(info, dt_sec)
            )
        elif source == "ball":
            if (
                self.config.enable_ball_lost_recovery
                and not self._is_detected_ball(info)
                and self.ball_tracking_active
            ):
                return self._lost_ball_recovery_command()
            command = (
                self.ball_planner.stop("waiting_for_ball_info")
                if info is None
                else self.ball_planner.plan(info, dt_sec)
            )
        elif source == "goal":
            if not self._is_detected_goal(info) and self.goal_tracking_active:
                return self._lost_goal_recovery_command()
            if self.goal_recovery_centering and self._is_detected_goal(info):
                return self._reacquired_goal_centering_command(info)
            command = (
                self.goal_planner.wait("waiting_for_goal_info")
                if info is None
                else self.goal_planner.plan(info)
            )
        else:
            command = (
                self.hurdle_planner.wait("waiting_for_hurdle_info")
                if info is None
                else self.hurdle_planner.plan(info)
            )
        return command.to_dict()

    @staticmethod
    def _number(data: dict[str, Any] | None, key: str) -> float | None:
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

    def _ball_range_m(self, info: dict[str, Any] | None) -> float | None:
        return self._number(info, "depth_m")

    @staticmethod
    def _is_detected_ball(info: dict[str, Any] | None) -> bool:
        return bool(info is not None and info.get("detected", False))

    def _ball_is_inside_control_range(
        self,
        info: dict[str, Any] | None,
    ) -> bool:
        """Allow confirmed RGB alignment while keeping depth-gated travel."""
        if not self._is_detected_ball(info):
            return False
        if not bool(info.get("depth_valid", False)):
            # BallNavigationPlanner can turn from steering_angle_deg without
            # depth, but it refuses forward travel until depth becomes valid.
            # Let a confirmed RGB ball preempt LINE so that alignment starts
            # instead of silently continuing along the line.
            return True
        ball_range = self._ball_range_m(info)
        return bool(
            ball_range is not None
            and ball_range <= self.config.ball_control_range_m
        )

    def _update_ball_tracking(
        self,
        info: dict[str, Any] | None,
        dt_sec: float,
    ) -> None:
        """Remember an in-range ball and time any later image loss."""
        if not self.config.enable_ball_lost_recovery:
            self._clear_ball_tracking()
            return
        if (
            info is not None
            and info.get("note") == "ball_outside_tracking_range"
            and not self.ball_lock_active
        ):
            self._clear_ball_tracking()
            return
        detected = self._is_detected_ball(info)
        confidence = self._number(info, "confidence")
        reliable = detected and confidence is not None and confidence >= 0.35

        if reliable:
            bearing = self._ball_direction_error_deg(info)
            offset = self._number(info, "offset_x_norm")
            if bearing is not None:
                self.last_ball_bearing_deg = bearing
            if offset is not None:
                self.last_ball_offset_x_norm = offset
            depth = self._ball_range_m(info)
            depth_valid = bool(info.get("depth_valid", False))
            if depth_valid and depth is not None:
                self.last_ball_depth_m = depth

            direction_value = bearing
            if direction_value is None and offset is not None:
                direction_value = offset * 35.0
            if direction_value is not None:
                deadband = self.config.ball_recovery_direction_deadband_deg
                if direction_value > deadband:
                    self.last_ball_turn_direction = "RIGHT"
                elif direction_value < -deadband:
                    self.last_ball_turn_direction = "LEFT"

            ball_range = self._ball_range_m(info)
            visual_alignment_only = not depth_valid
            if visual_alignment_only or (
                ball_range is not None
                and ball_range <= self.config.ball_control_range_m
            ):
                self.ball_tracking_active = True
            if self.ball_tracking_active:
                self.ball_lost_elapsed_sec = 0.0
                self.ball_recovery_timed_out = False
                # Detection recovery is complete as soon as a reliable ball
                # returns. The regular ball planner must then choose the
                # numbered TURN/STRAIGHT action from the current path angle.
                self.ball_recovery_centering = False
            return

        if not self.ball_tracking_active:
            return
        self.ball_recovery_centering = True
        self.ball_lost_elapsed_sec += max(0.0, dt_sec)
        if (
            self.ball_lost_elapsed_sec
            > self.config.ball_recovery_timeout_sec
        ):
            self.ball_recovery_timed_out = True

    def _lost_ball_recovery_command(self) -> dict[str, Any]:
        """Search from the last ball pose without returning to line mode."""
        elapsed = self.ball_lost_elapsed_sec
        stop_sec = self.config.ball_lost_stop_sec
        duration = self.config.ball_recovery_command_sec
        direction = self.last_ball_turn_direction
        motion = "BALL_LOST_STOP"
        reason = "ball_lost_stop_before_search"
        recovery_phase = "STOP"
        valid = True
        linear_speed = 0.0
        angular_speed = 0.0

        if self.ball_recovery_timed_out:
            reason = "ball_recovery_timeout"
            recovery_phase = "TIMEOUT"
            valid = False
        elif elapsed > stop_sec:
            search_elapsed = elapsed - stop_sec
            last_error = self._last_ball_direction_error_deg()
            initial_turn_sec = self._ball_initial_recovery_turn_sec(
                last_error
            )
            if search_elapsed <= initial_turn_sec:
                motion = f"RECOVER_TURN_{direction}"
                angular_speed = self._ball_recovery_turn_speed(direction)
                reason = "turn_toward_last_seen_ball_side"
                recovery_phase = "LAST_DIRECTION"
            else:
                search_elapsed -= initial_turn_sec
                forward_allowed = bool(
                    last_error is not None
                    and abs(last_error)
                    <= self.config.ball_recovery_forward_max_bearing_deg
                    and self.last_ball_depth_m is not None
                    and self.last_ball_depth_m
                    >= self.config.ball_recovery_forward_min_depth_m
                    and self.last_ball_depth_m
                    <= self.config.ball_control_range_m
                )
                forward_sec = (
                    self.config.ball_recovery_forward_sec
                    if forward_allowed
                    else 0.0
                )
                if search_elapsed <= forward_sec:
                    motion = "STRAIGHT_1"
                    linear_speed = 0.012
                    reason = "advance_toward_last_seen_ball"
                    recovery_phase = "FORWARD"
                else:
                    search_elapsed -= forward_sec
                    sweep_sec = max(
                        self.config.ball_recovery_sweep_sec,
                        duration,
                    )
                    sweep_index = int(search_elapsed // sweep_sec)
                    if sweep_index % 2 == 1:
                        direction = self._opposite_direction(direction)
                    motion = f"RECOVER_TURN_{direction}"
                    angular_speed = self._ball_recovery_turn_speed(direction)
                    reason = "alternating_ball_search"
                    recovery_phase = "SWEEP"

        return {
            "valid": valid,
            "motion": motion,
            "reason": reason,
            "linear_speed_mps": linear_speed,
            "lateral_speed_mps": 0.0,
            "angular_speed_rad_s": round(angular_speed, 4),
            "angular_accel_rad_s2": 0.0,
            "command_duration_sec": round(duration, 3),
            "travel_distance_m": round(linear_speed * duration, 4),
            "lateral_travel_distance_m": 0.0,
            "target_heading_change_deg": round(
                math.degrees(angular_speed * duration),
                3,
            ),
            "bearing_error_deg": self.last_ball_bearing_deg,
            "offset_x_norm": self.last_ball_offset_x_norm,
            "depth_m": None,
            "distance_m": None,
            "distance_error_m": None,
            "confidence": 0.0,
            "depth_valid": False,
            "pickup_ready": False,
            "pickup_now": False,
            "tracking_active": True,
            "lost_elapsed_sec": round(self.ball_lost_elapsed_sec, 3),
            "last_seen_direction": direction,
            "last_seen_depth_m": self.last_ball_depth_m,
            "recovery_phase": recovery_phase,
        }

    def _last_ball_direction_error_deg(self) -> float | None:
        """Return the last horizontal ball error in degrees."""
        if self.last_ball_bearing_deg is not None:
            return self.last_ball_bearing_deg
        if self.last_ball_offset_x_norm is None:
            return None
        return self.last_ball_offset_x_norm * 35.0

    def _ball_direction_error_deg(
        self,
        info: dict[str, Any] | None,
    ) -> float | None:
        """Prefer the calibrated bottom-center ball path angle."""
        steering = self._number(info, "steering_angle_deg")
        if steering is not None:
            return steering
        return self._number(info, "bearing_deg")

    def _ball_initial_recovery_turn_sec(
        self,
        error_deg: float | None,
    ) -> float:
        """Estimate a bounded turn time toward the last observed bearing."""
        if (
            error_deg is None
            or abs(error_deg) <= self.config.ball_reacquire_center_deg
        ):
            return 0.0
        turn_speed = max(abs(self.config.ball_recovery_turn_rad_s), 1e-3)
        estimated = abs(math.radians(error_deg)) / turn_speed
        return min(
            max(estimated, self.config.ball_recovery_command_sec),
            self.config.ball_recovery_initial_turn_max_sec,
        )

    def _ball_recovery_turn_speed(self, direction: str) -> float:
        """Return signed recovery yaw speed for one direction."""
        sign = 1.0 if direction == "RIGHT" else -1.0
        return sign * self.config.ball_recovery_turn_rad_s

    @staticmethod
    def _opposite_direction(direction: str) -> str:
        """Return the opposite horizontal search direction."""
        return "LEFT" if direction == "RIGHT" else "RIGHT"

    def _clear_ball_tracking(self) -> None:
        self.ball_tracking_active = False
        self.ball_recovery_centering = False
        self.ball_lost_elapsed_sec = 0.0
        self.last_ball_bearing_deg = None
        self.last_ball_offset_x_norm = None
        self.last_ball_depth_m = None
        self.ball_recovery_timed_out = False

    def ball_tracking_status(self) -> dict[str, Any]:
        """Expose remembered-ball state for debugging and behavior logs."""
        return {
            "active": self.ball_tracking_active,
            "recovery_centering": self.ball_recovery_centering,
            "tracking_range_m": self.config.ball_tracking_range_m,
            "control_range_m": self.config.ball_control_range_m,
            "lost_elapsed_sec": round(self.ball_lost_elapsed_sec, 3),
            "last_bearing_deg": self.last_ball_bearing_deg,
            "last_offset_x_norm": self.last_ball_offset_x_norm,
            "last_depth_m": self.last_ball_depth_m,
            "last_direction": self.last_ball_turn_direction,
            "timed_out": self.ball_recovery_timed_out,
        }

    @staticmethod
    def _is_detected_goal(info: dict[str, Any] | None) -> bool:
        return bool(info is not None and info.get("detected", False))

    def _goal_is_inside_control_range(
        self,
        info: dict[str, Any] | None,
    ) -> bool:
        if not self._is_detected_goal(info):
            return False
        depth = self._number(info, "depth_m")
        return bool(
            info.get("depth_valid", False)
            and depth is not None
            and depth <= self.config.goal_control_range_m
        )

    def _update_goal_tracking(
        self,
        info: dict[str, Any] | None,
        dt_sec: float,
    ) -> None:
        """Remember an in-range backboard and time any later image loss."""
        if (
            info is not None
            and info.get("note") == "goal_outside_tracking_range"
        ):
            self._clear_goal_tracking()
            return
        detected = self._is_detected_goal(info)
        confidence = self._number(info, "confidence")
        reliable = detected and confidence is not None and confidence >= 0.35

        if reliable:
            bearing = self._number(info, "bearing_deg")
            offset = self._number(info, "offset_x_norm")
            if bearing is not None:
                self.last_goal_bearing_deg = bearing
            if offset is not None:
                self.last_goal_offset_x_norm = offset

            direction_value = bearing
            if direction_value is None and offset is not None:
                direction_value = offset * 35.0
            if direction_value is not None:
                if direction_value > 1.0:
                    self.last_goal_turn_direction = "RIGHT"
                elif direction_value < -1.0:
                    self.last_goal_turn_direction = "LEFT"

            depth = self._number(info, "depth_m")
            if (
                bool(info.get("depth_valid", False))
                and depth is not None
                and depth <= self.config.goal_tracking_range_m
            ):
                self.goal_tracking_active = True
            if self.goal_tracking_active:
                self.goal_lost_elapsed_sec = 0.0
                if self.goal_recovery_centering and self._goal_is_centered(
                    info
                ):
                    self.goal_recovery_centering = False
            return

        if not self.goal_tracking_active:
            return
        self.goal_recovery_centering = True
        self.goal_lost_elapsed_sec += max(0.0, dt_sec)
        if (
            self.goal_lost_elapsed_sec
            > self.config.goal_recovery_timeout_sec
        ):
            self._clear_goal_tracking()

    def _lost_goal_recovery_command(self) -> dict[str, Any]:
        """Stop first, then rotate toward the last observed backboard side."""
        direction = self.last_goal_turn_direction
        stopping = (
            self.goal_lost_elapsed_sec <= self.config.goal_lost_stop_sec
        )
        if stopping:
            action = "GOAL_LOST_STOP"
            angular_speed = 0.0
            reason = "goal_lost_stop_before_search"
        else:
            action = f"RECOVER_GOAL_TURN_{direction}"
            sign = 1.0 if direction == "RIGHT" else -1.0
            angular_speed = sign * self.config.goal_recovery_turn_rad_s
            reason = "turn_toward_last_seen_goal_side"

        duration = self.config.goal_recovery_command_sec
        return {
            "valid": True,
            "action": action,
            "reason": reason,
            "sdk_motion_requested": False,
            "linear_speed_mps": 0.0,
            "angular_speed_rad_s": round(angular_speed, 4),
            "command_duration_sec": round(duration, 3),
            "target_heading_change_deg": round(
                math.degrees(angular_speed * duration),
                3,
            ),
            "confidence": 0.0,
            "depth_m": None,
            "distance_m": None,
            "depth_error_m": None,
            "bearing_error_deg": self.last_goal_bearing_deg,
            "offset_x_norm": self.last_goal_offset_x_norm,
            "is_centered": False,
            "depth_in_score_range": False,
            "score_now": False,
            "tracking_active": True,
            "lost_elapsed_sec": round(self.goal_lost_elapsed_sec, 3),
            "last_seen_direction": direction,
        }

    def _goal_is_centered(self, info: dict[str, Any] | None) -> bool:
        bearing = self._number(info, "bearing_deg")
        if bearing is not None:
            return abs(bearing) <= self.config.goal_reacquire_center_deg
        offset = self._number(info, "offset_x_norm")
        return bool(
            offset is not None
            and abs(offset) <= self.config.goal_reacquire_center_norm
        )

    def _reacquired_goal_centering_command(
        self,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep rotating after goal reacquisition until it is centered."""
        bearing = self._number(info, "bearing_deg")
        offset = self._number(info, "offset_x_norm")
        direction_value = bearing
        if direction_value is None and offset is not None:
            direction_value = offset * 35.0
        if direction_value is not None and direction_value < 0.0:
            direction = "LEFT"
            sign = -1.0
        else:
            direction = "RIGHT"
            sign = 1.0
        angular_speed = sign * self.config.goal_recovery_turn_rad_s
        duration = self.config.goal_recovery_command_sec
        return {
            "valid": True,
            "action": f"RECOVER_GOAL_TURN_{direction}",
            "reason": "reacquired_goal_centering_in_place",
            "sdk_motion_requested": False,
            "linear_speed_mps": 0.0,
            "angular_speed_rad_s": round(angular_speed, 4),
            "command_duration_sec": round(duration, 3),
            "target_heading_change_deg": round(
                math.degrees(angular_speed * duration),
                3,
            ),
            "confidence": self._number(info, "confidence") or 0.0,
            "depth_m": self._number(info, "depth_m"),
            "distance_m": self._number(info, "distance_m"),
            "depth_error_m": None,
            "bearing_error_deg": bearing,
            "offset_x_norm": offset,
            "is_centered": False,
            "depth_in_score_range": False,
            "score_now": False,
            "tracking_active": True,
            "lost_elapsed_sec": 0.0,
            "last_seen_direction": direction,
        }

    def _clear_goal_tracking(self) -> None:
        self.goal_tracking_active = False
        self.goal_recovery_centering = False
        self.goal_lost_elapsed_sec = 0.0
        self.last_goal_bearing_deg = None
        self.last_goal_offset_x_norm = None

    def goal_tracking_status(self) -> dict[str, Any]:
        """Expose remembered-goal state for debugging and behavior logs."""
        return {
            "active": self.goal_tracking_active,
            "recovery_centering": self.goal_recovery_centering,
            "tracking_range_m": self.config.goal_tracking_range_m,
            "control_range_m": self.config.goal_control_range_m,
            "lost_elapsed_sec": round(self.goal_lost_elapsed_sec, 3),
            "last_bearing_deg": self.last_goal_bearing_deg,
            "last_offset_x_norm": self.last_goal_offset_x_norm,
            "last_direction": self.last_goal_turn_direction,
        }

    def _reset_source(self, source: str) -> None:
        if source == "line":
            self.line_planner.stop("source_changed")
        elif source == "ball":
            self.ball_planner.stop("source_changed")

    def _reset_previous_source(self) -> None:
        self._reset_source(self.previous_source)
        self.previous_source = "none"
