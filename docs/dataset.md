# 데이터셋 및 모델

## 데이터셋 형식

두 가지 레이아웃 지원. `--data-layout auto`(기본)로 자동 판별.

**Layout A** — 단일 영상 (이미지/라벨이 한 폴더에):
```
dataset_root/
├── images/       # 입력 이미지 (.jpg, .png)
├── labels/       # 2.5D 라벨 (txt)
└── calib/        # 3x3 투영행렬 — 추론 시에만 사용
```

**Layout B** — 다중 영상 (영상별 하위 폴더):
```
dataset_root/
├── video_001/
│   ├── images/
│   └── labels/
├── video_002/
│   ├── images/
│   └── labels/
└── ...
```

> 학습: `images/`, `labels/`만 필요. `calib/`는 추론 시에만 사용.

### 시퀀스 학습 (ConvLSTM)

ConvLSTM은 **연속 프레임**의 시간적 맥락을 학습. 이미지 파일명이 시간순 정렬 가능해야 함.

- **Layout B**: 영상별 하위 폴더 단위로 시퀀스를 자동 구성
- **Layout A**: 파일명 prefix (`video01_000.jpg`, `video01_001.jpg`, ...)로 영상을 구분하여 시퀀스 구성
- `--seq-len` 프레임 미만인 영상은 학습에서 제외

### 라벨 형식 (space 구분)

**학습/추론 라벨** — 이미지 좌표 3점 (7열):
```
class P0x P0y P1x P1y P2x P2y
```
- P0: 차량 바닥 중심점, P1/P2: 전면 바닥 꼭짓점 (순서 무관)
- 모든 좌표는 이미지 픽셀 좌표

**BEV 라벨** — 투영행렬/LUT로 변환한 월드 좌표:
```
class cx cy length width yaw_deg                          # 6열
class cx cy cz length width yaw_deg pitch_deg roll_deg    # 9열
```

## 제공 모델

| 모델 | 학습 데이터 | 설명 |
|------|------------|------|
| `carla_base` | CARLA 시뮬레이션 ~20만장 (다양한 각도/높이) | 기본 모델. 단일클래스 |
| `carla_pointcloud` | carla_base + CARLA 자체 제작 차량/맵 데이터로 파인튜닝 | LUT(npz) 방식용. 단일클래스 |
| `ces_real` | carla_base + 실환경 CES 시연 영상 ~3000장 수동 라벨링으로 파인튜닝 | Sim2Real. 멀티클래스 |

> 모든 모델은 `onnx/`에 ONNX 파일로 제공. `pth/`에는 carla_base 체크포인트(resume 가능)만 포함.

### 성능 지표 (dataset_example 기준)

| 모델 | conf | Precision | Recall | mAP@50 | mAOE(°) |
|------|------|-----------|--------|--------|---------|
| `carla_base` | 0.8 | 1.0000 | 0.9244 | 0.9244 | 0.78 |
| `carla_pointcloud` | 0.8 | 0.9231 | 0.8000 | 0.7557 | 0.91 |
| `ces_real` | 0:0.1 / 1:0.8 / 2:0.8 | 0.9085 | 0.8613 | 0.8510 | 2.75 |

> 각 예제 데이터셋(30프레임)에 대한 평가 결과. mAOE = 평균 방향각 오차(낮을수록 좋음).

**ces_real 클래스별 성능 (2D)**:

| 클래스 | Precision | Recall | AP@50 | mAOE(°) |
|--------|-----------|--------|-------|---------|
| 0: 차량 | 0.7973 | 0.7108 | 0.6593 | 3.61 |
| 1: 바리케이드 | 1.0000 | 1.0000 | 1.0000 | 2.66 |
| 2: 라바콘 | 1.0000 | 1.0000 | 1.0000 | 1.25 |

## dataset_example/

| 디렉토리 | BEV 방식 | 내용 |
|----------|----------|------|
| `carla_base/` | 투영행렬(txt) | images, labels, calib |
| `carla_pointcloud/` | LUT(npz) | images, labels, npz, ply |
| `ces_real/` | 투영행렬(txt) | images, labels, calib |
