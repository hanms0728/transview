# 학습 및 추론 코드 상세

## 모델 구조

YOLO11m의 backbone + FPN은 그대로 가져오고, detection head만 제거하여 2.5D 예측용 커스텀 헤드(TriHead)로 교체한 구조.

### Backbone — YOLO11m + FPN

Ultralytics YOLO11m의 backbone과 neck(FPN)에서 멀티스케일 feature map 추출.

- **P3**(stride 8), **P4**(stride 16), **P5**(stride 32)
- `--p2-head` 옵션으로 P2(stride 4) 추가 가능 — 소형 객체 대응

### Temporal Block — ConvLSTM

P5 feature map에 ConvLSTM을 적용하여 프레임 간 시간적 맥락을 학습.

```
Input feature (C ch)
    → 1×1 Conv (C → 256 ch, 채널 축소)
    → ConvLSTM Cell (3×3 kernel, hidden state 유지)
    → 1×1 Conv (256 → C ch, 채널 복원)
Output feature (C ch)
```

- Bottleneck 구조로 연산량 절감
- Hidden state는 같은 영상 시퀀스 내에서 프레임 간 전달 (streaming mode)
- TBPTT(Truncated Backpropagation Through Time): 매 타임스텝마다 state를 detach하여 메모리 안정화

### 2.5D TriHead

각 스케일(P3/P4/P5)마다 독립적인 예측 헤드가 3개 브랜치를 동시에 출력:

| 브랜치 | 채널 수 | 출력 |
|--------|---------|------|
| Regression | 6 | 3점 × 2좌표(x, y) — 그리드 셀 기준 이미지 좌표 오프셋 |
| Objectness | 1 | 물체 존재 확률 |
| Classification | N | 클래스별 확률 |

**Regression 6채널의 의미:**

모델이 예측하는 3점은 모두 차량 바닥면 기준의 이미지 좌표:
- **P0** — 차량 바닥 중심점
- **P1, P2** — 차량 전면 바닥 꼭짓점 2개 (순서 무관)

```
        P1
       / |
      /  |        ← 차량 바닥면을 위에서 본 형태
  P0 ·   |           P0~P2 세 점으로 위치·크기·방향 결정
      \  |
       \ |
        P2
```

이 3점이 곧 차량의 위치(P0), 방향(P0→P1,P2의 중점), 폭(P1↔P2 거리), 길이(P0↔전면 중점 거리 × 2)를 결정한다. 추론 시 이 이미지 좌표 3점을 BEV 변환(homography/LUT)하여 월드 좌표계의 cx, cy, length, width, yaw로 변환.

## 라벨 어사인

삼각형 기반 + 거리 기반 하이브리드 방식으로 GT를 그리드 셀에 할당:

1. GT 삼각형(center + front corners 2개) 내부에 있는 셀 → **positive**
2. 삼각형 외부이지만 가장자리에서 `eta_px`(기본 3.0px) 이내인 셀 → **positive**
3. 나머지 → **negative**
4. positive 셀이 `k_pos_cap`(기본 96)개를 초과하면, 가장 가까운 셀만 선택

## Loss 함수

### Regression Loss
- **P0 (중심점):** MSE Loss
- **P1, P2 (전면 꼭짓점):** Chamfer 2-point distance
- 두 loss를 `lambda_p0`, `lambda_cd` 가중치로 합산

### Objectness Loss
- Binary Cross Entropy with Logits
- positive/negative 셀 수로 정규화

### Classification Loss
- 단일 클래스: BCE
- 멀티클래스: Cross Entropy + Focal Loss (선택)
  - Focal gamma가 0보다 클 때: `(1 - pt)^γ × CE`
  - 클래스별 가중치 설정 가능 (`--cls-weights`)

### DSI (Deep Structural Inference)

Pretrained ViT(Vision Transformer) teacher로부터 student backbone의 feature를 정규화하는 보조 loss.

- **Teacher:** ViT-B/16 (ImageNet SWAG pretrained, 학습 중 frozen)
- **방식:** Student의 deep feature를 1×1 conv로 프로젝션 → teacher feature와 cosine similarity 최대화
- **GAM (Gradient-based Adaptive Mixing):** backbone/head의 gradient 비율을 모니터링하여 DSI 가중치를 자동 조절
- 학습 중에만 사용, 추론 시에는 비활성

## 학습 전략

### Optimizer
- **Backbone:** SGD (lr=2e-4, momentum=0.9, Nesterov)
- **Head + Temporal:** AdamW (lr=1e-3)
- 각각 별도 CosineAnnealingLR 스케줄러 적용 (eta_min=1e-4)

### Backbone Freeze
- 초기 N 에포크(`--freeze-bb-epochs`, 기본 1) 동안 backbone 가중치를 고정
- Head/Temporal이 안정된 후 backbone fine-tuning 시작

### 시퀀스 학습
- `SeqWindowDataset`이 연속 프레임을 T개 단위 윈도우로 묶음
- Streaming mode: 같은 영상에서 온 배치는 hidden state 유지, 다른 영상이면 reset
- `--seq-len 4 --seq-stride 1`로 시퀀스 수 최대화 가능

### 안정화 기법
- **AMP (Automatic Mixed Precision):** FP16 연산 + GradScaler
- **TF32:** `torch.backends.cuda.matmul.allow_tf32 = True`
- **Gradient Clipping:** max_norm=10.0
- **Gradient Sanitization:** NaN/Inf gradient를 0으로 치환
- **Bad Batch Skip:** loss가 NaN이면 해당 배치 건너뜀

## Data Augmentation

Albumentations 기반 (`--train-augment`):

| 증강 | 파라미터 | 확률 |
|------|---------|------|
| RandomResizedCrop | scale 0.85~1.0 | 0.4 |
| HorizontalFlip | — | 0.5 |
| RandomBrightnessContrast | ±0.2 | 0.5 |
| ColorJitter | brightness/contrast/saturation ±0.2, hue ±0.02 | 0.4 |
| Affine | scale 0.9~1.1, rotate ±5°, translate ±2% | 0.6 |
| Perspective | scale 0.02~0.05 | 0.3 |
| GaussNoise | var 5~30 | 0.3 |

## 검증 및 평가

- **메트릭:** mAP@50, mAOE(평균 방향 오차), 클래스별 precision/recall
- **Score mode:** `obj × cls` (기본), `obj`, `cls` 선택 가능
- **NMS IoU:** 0.20 (기본)
- **Best 모델:** mAP@50 기준 자동 저장

## ONNX Export

매 에포크마다 자동 export (opset 17):

| Temporal 모드 | 입력 | 출력 |
|--------------|------|------|
| none | image | predictions |
| lstm | image + h_in + c_in | predictions + h_out + c_out |

- `temporal_on_scales='last'`만 지원
- Dynamic batch axes 설정
