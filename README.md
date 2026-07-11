# IRC Vision

ROS 2 카메라 영상을 OpenCV로 처리하여 바닥의 흰색 테이프를 검출하고, 로봇 기준 진행 방향과 조향 각도를 시각화하는 비전 프로젝트입니다.

## 동작 방식

노드는 `/camera/camera/color/image_raw` 토픽을 구독하고 다음 과정을 수행합니다.

1. ROS 이미지를 OpenCV BGR 이미지로 변환
2. 관심 영역(ROI) 설정 및 그레이스케일 변환
3. 이진화와 모폴로지 연산으로 노이즈 제거
4. 윤곽선의 면적, 종횡비, 꼭짓점 수를 이용해 흰색 테이프 후보 필터링
5. 검출된 테이프의 중심과 각도를 계산하여 화면에 표시

## 프로젝트 구조

```text
.
└── src/step/
    ├── package.xml
    ├── setup.py
    └── step/
        ├── look_ground.py
        ├── look_gground.py
        ├── find_direct.py
        └── find_ddirect.py
```

각 실행 파일은 검출 범위와 방향 계산 방식이 조금씩 다른 실험용 구현입니다.

- `look_ground`: 전체 영상에서 테이프 중심과 각도 검출
- `look_gground`: 동적 ROI를 적용한 테이프 검출
- `find_direct`: 여러 테이프 중심의 평균을 이용한 진행 방향 계산
- `find_ddirect`: 테이프 중심 경로와 상대 각도를 이용한 진행 방향 계산

## 요구 사항

- ROS 2
- Python 3
- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- OpenCV
- NumPy
- 컬러 이미지 토픽을 제공하는 카메라 노드

Intel RealSense를 사용하는 경우 `realsense2_camera` 노드를 실행하여 `/camera/camera/color/image_raw` 토픽을 제공할 수 있습니다.

## 빌드

저장소 루트에서 ROS 2 환경을 불러오고 패키지를 빌드합니다.

```bash
source /opt/ros/<ros-distro>/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`<ros-distro>`는 설치한 ROS 2 배포판 이름으로 바꾸세요.

## 실행

먼저 카메라 노드를 실행한 뒤, 별도 터미널에서 원하는 비전 노드를 실행합니다.

```bash
source /opt/ros/<ros-distro>/setup.bash
source install/setup.bash
ros2 run step find_ddirect
```

다른 구현도 같은 방식으로 실행할 수 있습니다.

```bash
ros2 run step look_ground
ros2 run step look_gground
ros2 run step find_direct
ros2 run step find_ddirect
```

## 조정 가능한 값

조명과 카메라 위치에 따라 각 Python 파일의 다음 값을 조정해야 할 수 있습니다.

- 이진화 임계값
- ROI 비율
- 윤곽선 최소/최대 면적
- 테이프 종횡비 범위
- 좌회전/직진/우회전 판정 각도

현재 코드는 검출 결과와 방향을 OpenCV 창에 시각화합니다. 로봇 구동부에 제어 명령을 발행하는 기능은 포함되어 있지 않습니다.
