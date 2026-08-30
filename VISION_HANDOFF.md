# Vision handoff

이 변경의 소유 범위는 카메라 입력, YOLO 검출, 객체별 분석 JSON, 화면
표시까지입니다. 실제 모션 우선순위와 상태 전환은 Algorithm 쪽 소유이며
Vision은 `/navigation/motion_command`를 발행하지 않습니다.

## 실행

알고리즘 없이 Vision만 실행합니다.

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select step
source install/setup.bash
ros2 launch step vision_system.launch.py
```

카메라가 이미 실행 중이면 `enable_camera:=false`를 추가합니다.

## Algorithm이 읽을 토픽

- `/vision/detections`: 원시 YOLO bbox와 confidence
- `/vision/line_info`: 라인 기하 정보
- `/vision/ball_info`: 공의 bbox, 조향각, depth와 바닥면 거리
- `/vision/hurdle_info`: 허들의 bbox, 정렬값, depth와 바닥면 거리
- `/vision/goal_info`: 골대의 bbox, 조준점, depth와 바닥면 거리

객체 JSON의 `detected`는 RGB 검출·확정 상태입니다. 전진, 집기, 슛,
허들 넘기처럼 거리가 필요한 동작은 반드시 `depth_valid == true`를 함께
확인해야 합니다. `depth_m`은 카메라 광축 거리이고
`ground_distance_m`은 카메라 하향각을 보정한 바닥면 수평거리입니다.

공의 `steering_angle_deg`는 영상 하단의 보정된 로봇 중심점과 공 중심을
이은 선의 각도입니다. 기본 로봇 중심선은 영상 중앙보다 오른쪽 70px이며
`robot_center_offset_px`로 조정할 수 있습니다.

화면의 `Suggested` 및 `BALL/GOAL/HURDLE ...` 배너는 확인용일 뿐입니다.
이 값은 ROS 토픽으로 모션을 발행하지 않으며 Algorithm 입력 계약에
포함되지 않습니다.

## 가져갈 런타임 파일

Algorithm 쪽에서는 아래 Vision 파일만 반영하면 됩니다.

```text
src/step/setup.py
src/step/launch/vision_system.launch.py
src/step/launch/ball_only_debug.launch.py
src/step/step/yolo26_detector.py
src/step/step/unified_vision_node.py
src/step/step/ball_analyzer.py
src/step/step/hurdle_analyzer.py
src/step/step/goal_analyzer.py
src/step/step/ball_only_debug.py
src/step/step/visual_motion_advisor.py
```

`mission_control`, `motion_decision*`, `*_navigation_planner.py`,
`*_navigation_controller.py`는 이 Vision handoff 대상이 아닙니다.
