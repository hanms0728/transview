# 사용법

## 학습

```bash
# 처음부터 학습
python -m src.train \
    --train-root <데이터셋 경로> \
    --temporal lstm --seq-len 4

# 이어서 학습 (체크포인트에서 재개)
python -m src.train \
    --train-root <데이터셋 경로> \
    --temporal lstm --seq-len 4 \
    --resume <체크포인트 경로>
```

결과 저장: `./results/train` (모델 가중치, ONNX, 학습 로그)

> `--resume`으로 이어서 학습 시 `.pth` 체크포인트 필요. `pth/carla_base.pth`는 CARLA 시뮬레이션 ~20만장으로 사전학습된 가중치로, 파인튜닝 시 베이스로 사용.

<details>
<summary>예제</summary>

```bash
python -m src.train \
    --train-root ./dataset_example/carla_base \
    --temporal lstm --seq-len 4
```

</details>

<details>
<summary>주요 옵션</summary>

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--temporal` | 시계열 모듈 (`none`, `lstm`, `gru`) | `lstm` |
| `--seq-len` | 시퀀스 길이 (ConvRNN은 4 이상 권장) | `4` |
| `--num-classes` | 클래스 수 | `1` |
| `--epochs` | 학습 에포크 | `60` |
| `--batch` | 배치 크기 | `4` |
| `--img-h`, `--img-w` | 입력 이미지 크기 | `864`, `1536` |
| `--yolo-weights` | YOLO11 백본 가중치 | `yolo11m.pt` |
| `--dsi` | Deep Structural Inference | `True` |
| `--save-dir` | 저장 경로 | `./results/train` |

</details>

## 추론

```bash
# 투영행렬(txt) 방식 — 기본
python -m src.inference \
    --input-dir <이미지 경로> \
    --weights <ONNX 모델 경로> \
    --calib-dir <3x3 투영행렬 경로>

# LUT(npz) 방식
python -m src.inference \
    --input-dir <이미지 경로> \
    --weights <ONNX 모델 경로> \
    --bev-mode lut --lut-path <LUT npz 경로>
```

결과 저장: `./results/inference` (2D 이미지, BEV 이미지, 라벨)

> `--gt-label-dir` 지정 시 mAP, mAOE 등 평가 지표 출력. 생략 시 추론만 수행.

<details>
<summary>예제 (3가지 시나리오)</summary>

```bash
# 1. CARLA 기본 — 투영행렬(txt) 방식 (기본)
python -m src.inference \
    --input-dir ./dataset_example/carla_base/images \
    --weights ./onnx/carla_base.onnx \
    --calib-dir ./dataset_example/carla_base/calib

# 2. CES 실환경 (Sim2Real) — 투영행렬(txt) 방식
python -m src.inference \
    --input-dir ./dataset_example/ces_real/images \
    --weights ./onnx/ces_real.onnx \
    --calib-dir ./dataset_example/ces_real/calib

# 3. CARLA 포인트클라우드 — LUT(npz) 방식
python -m src.inference \
    --input-dir ./dataset_example/carla_pointcloud/images \
    --weights ./onnx/carla_pointcloud.onnx \
    --bev-mode lut \
    --lut-path ./dataset_example/carla_pointcloud/carla_pointcloud.npz
```

</details>

<details>
<summary>주요 옵션</summary>

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--bev-mode` | BEV 변환 방식 (`homography`, `lut`) | `homography` |
| `--calib-dir` | 3x3 투영행렬 디렉토리 (homography 모드 시) | `None` |
| `--lut-path` | LUT npz 파일 경로 (lut 모드 시) | `None` |
| `--conf` | 신뢰도 임계값 | `0.30` |
| `--class-conf-map` | 클래스별 임계값 (예: `"0:0.7,1:0.5"`) | `None` |
| `--allowed-classes` | 허용할 클래스 ID (예: `"0,1"`) | `None` |
| `--temporal` | 시계열 모드 (`none`, `lstm`, `gru`) | `lstm` |
| `--gt-label-dir` | GT 라벨 경로 (평가 시) | `None` |
| `--output-dir` | 결과 저장 경로 | `./results/inference` |
| `--no-cuda` | CPU 모드 | `False` |

</details>

데이터셋 형식, 라벨 형식, 제공 모델/데이터셋 정보는 [dataset.md](dataset.md) 참고.
