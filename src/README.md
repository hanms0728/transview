# YOLO11 2.5D Polygon Trainer (`src/train.py`)

`src/train.py`는 Ultralytics YOLO11 백본 위에 삼각형 기반 2.5D 헤드를 얹어 차량 지붕/물체 평행사변형을 학습하는 스크립트입니다. 멀티 클래스, 혼합 정밀도, 자동 체크포인트/ONNX 내보내기, 선택적 데이터 증강과 Focal Loss 등을 모두 포함한 완전 버전입니다.

---

## 1. 데이터 준비

라벨 파일은 YOLO 스타일의 텍스트 (`class p0x p0y p1x p1y p2x p2y`) 형식이어야 하며, 아래 두 가지 디렉터리 구조 중 하나를 따르면 됩니다.

### A. Flat 구조
```
/dataset_root
 ├── images/
 │    ├── 0001.jpg
 │    └── 0002.jpg
 └── labels/
      ├── 0001.txt   # class p0x p0y p1x p1y p2x p2y
      └── 0002.txt
```

### B. 멀티 폴더 구조 (카메라/시퀀스별 서브폴더)
```
/dataset_root
 ├── cam01/
 │    ├── images/
 │    └── labels/
 ├── cam02/
 │    ├── images/
 │    └── labels/
 └── ...
```

`--train-root`/`--val-root`를 각각 위 구조의 최상위 폴더로 넘기면 스크립트가 자동으로 감지합니다.

---

## 2. 주요 기능

- **멀티 클래스**: 라벨 첫 열(class id)을 그대로 사용하며, `--num-classes`로 모델/손실을 설정합니다.
- **커스텀 손실**: 삼각형 오프셋 + Chamfer distance 기반 `Strict2_5DLoss`, 선택적 분류 Focal Loss(`--cls-focal`).
- **데이터 증강**: `--train-augment` 플래그를 켜면 Albumentations 기반 flip/brightness/crop/perspective/noise 를 적용합니다.
- **검증/로깅**: mAP/precision/recall/mAOE + 클래스별 지표, epoch마다 PTH/CKPT/CSV 저장, 옵션에 따라 ONNX 내보내기.

---

## 3. 실행 예시

### 3.1 단일 클래스 (차량만) 학습
```bash
python -m src.train \
  --train-root /data/car_only/train \
  --val-root /data/car_only/val \
  --yolo-weights yolo11m.pt \
  --save-dir ./runs/car_only \
  --epochs 60 --batch 4 --img-h 864 --img-w 1536
```
- 기본값이 1클래스이므로 `--num-classes`를 줄 필요가 없습니다.
- 증강을 쓰고 싶다면 `--train-augment`를 추가하세요.

### 3.2 멀티 클래스 (예: 차량/콘/박스) 학습
```bash
python -m src.train \
  --train-root /data/multi/train \
  --val-root /data/multi/val \
  --yolo-weights yolo11m.pt \
  --save-dir ./runs/multi_cls \
  --num-classes 3 \
  --cls-focal --focal-gamma 2.0 --focal-alpha 0.25 \
  --train-augment \
  --epochs 80 --batch 4 --img-h 864 --img-w 1536
```
- 라벨의 클래스 ID 범위(0~2)에 맞춰 `--num-classes 3` 지정.
- 소수 클래스 대응을 위해 Focal Loss를 권장합니다.

---

## 4. 주요 CLI 옵션

| 옵션 | 설명 |
| --- | --- |
| `--train-root`, `--val-root` | 데이터셋 경로 (Flat/멀티폴더 자동 감지). `--val-root`를 생략하면 train과 같은 경로에서 `images/labels` 사용 |
| `--yolo-weights` | 초기 YOLO11 가중치 (`yolo11m.pt` 등) |
| `--num-classes` | 클래스 수 (기본 1). 라벨의 최대 class id + 1과 일치시켜야 함 |
| `--train-augment` | Albumentations 증강을 켭니다 (설치 필요) |
| `--cls-focal`, `--focal-gamma`, `--focal-alpha` | 분류 헤드에 Focal Loss 적용 및 파라미터 조정 |
| `--onnx-dir`, `--onnx-opset` | ONNX 저장 경로/OPSET 설정 (매 epoch + best 모델을 자동 저장) |
| 기타 | 배치/에폭/러닝레이트/Freeze epoch 등은 기본 argparse 옵션 참조 |

> **주의**: `--train-augment`를 사용할 경우 `pip install albumentations opencv-python`이 되어 있어야 합니다.

---

## 5. 결과물

`--save-dir` (기본 `./runs/2p5d`) 아래에 다음이 생성됩니다.

- `pth/`: 매 epoch마다 저장되는 PyTorch 가중치 + `*_best.pth`.
- `ckpt/`: 옵티마/스케줄러/Scaler가 포함된 체크포인트.
- `*.csv`: 에폭별 손실/지표 로그.
- `onnx/`: 각 epoch 및 best 모델의 ONNX 내보내기 결과.

---

## 6. 기타 팁

- 증강과 Focal Loss는 옵션이며, 켠/끈 버전을 각각 짧게 검증해 가장 안정적인 조합을 선택하세요.
- 멀티 클래스일 때 검증 로그에 `[clsN: P=..., R=..., mAP=...]` 형태로 클래스별 지표가 출력되므로, 특정 클래스가 부족한지 쉽게 확인할 수 있습니다.
- 라벨 포맷이 잘못되거나 짝이 맞지 않으면 스크립트가 자동으로 해당 샘플을 건너뜁니다. 로그를 통해 매칭된 샘플 수를 확인하세요.

---

# YOLO11 2.5D LSTM Trainer (`src/train_lstm_onnx.py`)

LSTM/ConvRNN 기반 시계열 헤드를 사용해 프레임 시퀀스(T≥2)를 학습하는 전체 파이프라인입니다. `src/train.py`와 동일한 데이터 구조/라벨 포맷을 따르되, 시퀀스 윈도우(`--seq-len`, `--seq-stride`)를 만들어 LSTM hidden state를 함께 최적화합니다.

## 1. 필수 개념

- **데이터 구조**: `ParallelogramDataset`/`SeqWindowDataset`은 위에서 설명한 A/B 구조를 그대로 사용합니다. `--seq-grouping auto`는 라벨 이름 프리픽스/서브폴더를 기준으로 시퀀스를 자동 생성합니다.
- **Temporal 모듈**: `--temporal lstm` 또는 `--temporal gru`로 헤드 앞에 ConvLSTM/ConvGRU를 붙일 수 있으며, `--temporal-on-scales`를 `last`로 두면 가장 깊은 피처 맵에만 적용합니다.
- **TBPTT**: `--tbptt-detach`를 켜면 각 타임스텝마다 hidden state를 끊어 그래프 폭주를 방지합니다. 시퀀스 길이가 짧고 GPU 메모리가 여유 있을 때만 끄고 전체 그래프를 역전파하세요.
- **DSI (Deep Semantic Injection)**: teacher(ViT) 피처에 student 피처를 정렬시키는 보조 손실입니다. 학습 중에만 추가되고 검증/추론에는 쓰이지 않으므로, 활성화 여부에 따라 train/val 분포 차이가 발생할 수 있습니다.
- **평가 스코어**: 기본은 `score = obj * cls`이지만 멀티 클래스 헤드를 새로 학습할 때는 `--score-mode obj`로 잠시 완화하면 cls 로그가 자리 잡을 때까지 mAP가 과도하게 흔들리지 않습니다.

## 2. 두 단계 학습 워크플로

소량 데이터(예: 100쌍)로 파이프라인을 검증한 뒤, 대량 데이터(예: 1,000장 이상)로 본 학습을 진행하는 단계적 접근을 권장합니다.

### Stage 1: 안정화 (증강/DSI 비활성화)

1. **목적**: 새 클래스 수(`--num-classes N`)나 LSTM 헤드 등이 정상 동작하는지 확인하고, cls 헤드가 랜덤 초기화된 상태에서 크게 흐트러지지 않도록 학습률을 낮게 두고 워밍업합니다.
2. **권장 하이퍼**:
   - `--batch 4` (가능하면 8 이상) / `--val-batch`도 동일하거나 근접하게 설정.
   - `--lr-bb 5e-5`, `--lr-hd 2e-4`, `--lr-min 5e-5`, `--freeze-bb-epochs 5`.
   - `--score-mode obj`, `--eval-conf 0.35` (cls가 안정되면 `obj*cls`로 되돌림).
   - `--no-train-augment --no-dsi`, `--skip-bad-batch --max-grad-norm 10.0`.
3. **실행 예시** (100장 셋으로 sanity check):

   ```bash
   python -m src.train_lstm_onnx \
     --train-root /media/ubuntu24/T7/val_dataset_lstm/cam2_-30 \
     --val-root   /media/ubuntu24/T7/val_dataset_lstm/cam2_-30 \
     --data-layout auto \
     --yolo-weights yolo11m.pt \
     --weights onnx/base_pth/yolo11m_2_5d_epoch_005.pth \
     --temporal lstm --temporal-hidden 256 --temporal-layers 1 --temporal-on-scales last \
     --seq-len 6 --seq-stride 2 --seq-grouping auto \
     --batch 4 --val-batch 4 \
     --start-epoch 5 \
     --no-seq-streaming --tbptt-detach --temporal-reset-per-batch \
     --freeze-bb-epochs 5 \
     --lr-bb 5e-5 --lr-hd 2e-4 --lr-min 5e-5 \
     --num-classes 3 \
     --score-mode obj --eval-conf 0.35 \
     --no-train-augment --no-dsi \
     --skip-bad-batch --max-grad-norm 10.0 \
     --save-dir runs/1208_lstm_stage1
   ```

4. **로그 해석**:
   - train loss가 완만히 감소하고 `Val`에서 precision/recall/mAP이 꾸준히 상승하면 Stage 1 OK.
   - cls 로짓이 안정되어 `score-mode obj*cls`로 바꿔도 mAP가 유지되면 Stage 2로 넘어갈 준비가 된 것입니다.

### Stage 2: 본 학습 (대량 데이터 + 증강/DSI)

1. **데이터 구성**:
   - `--train-root`와 `--val-root`를 서로 다른 시퀀스/카메라/날짜로 분리해 실제 일반화 성능을 측정합니다.
   - 1,000장 이상 데이터가 있으면 `--seq-stride 1`이나 더 작은 값을 고려해 시퀀스 수를 최대화하세요.
2. **체크포인트 이월**:
   - Stage 1 best PTH를 `--weights`에 주고 `--start-epoch 0`부터 다시 학습합니다.
   - 필요 시 `--resume`으로 옵티마/스케줄러 상태까지 복구할 수 있습니다.
3. **권장 설정**:
   - 배치를 가능한 한 크게(`batch 4~8`) 두고, GPU 여유가 없으면 `--seq-stride`를 조절해 데이터 양을 확보합니다.
   - 초반 몇 epoch은 Stage 1과 동일한 하이퍼로 돌린 뒤, `--train-augment`를 켜고 `--lr-hd`를 소폭 조정합니다.
   - DSI를 재사용하려면 `--dsi`(기본 ON)를 켜고, `--dsi-weight` 및 `--dsi-gam-*` 파라미터로 비중을 조절합니다. train/val 분포가 크게 달라졌다면 DSI를 다시 끄고 비교 실험을 진행하세요.
4. **예시 명령어 (Stage 2)**:

   ```bash
   python -m src.train_lstm_onnx \
     --train-root /data/full/train \
     --val-root   /data/full/val \
     --data-layout auto \
     --yolo-weights yolo11m.pt \
     --weights runs/1208_lstm_stage1/pth/yolo11m_2_5d_best.pth \
     --temporal lstm --temporal-hidden 256 --temporal-layers 1 --temporal-on-scales last \
     --seq-len 6 --seq-stride 1 --seq-grouping by_prefix \
     --batch 6 --val-batch 4 \
     --no-seq-streaming --tbptt-detach --temporal-reset-per-batch \
     --freeze-bb-epochs 3 \
     --lr-bb 7e-5 --lr-hd 3e-4 --lr-min 7e-5 \
     --num-classes 3 \
     --score-mode obj*cls --eval-conf 0.45 \
     --train-augment --dsi \
     --skip-bad-batch --max-grad-norm 10.0 \
     --save-dir runs/1208_lstm_stage2
   ```

   > 위 학습률/배치는 예시일 뿐이며, GPU 메모리·데이터 규모·클래스 수에 맞춰 조정하세요.

## 3. 하이퍼파라미터 가이드

- **Batch size**: 최소 2 이상을 권장합니다. `batch=1`은 gradient noise가 커서 손실은 줄어도 mAP가 크게 출렁입니다.
- **학습률**: warm-start 상태에서 새로운 클래스를 학습할 때는 backbone/head 학습률을 기본값 대비 3~5배 낮추고, CosineAnnealing 최소값(`--lr-min`)도 함께 낮춰야 안정적입니다.
- **Backbone freeze**: `--freeze-bb-epochs`를 3~5 이상으로 두고, cls가 자리 잡은 뒤 서서히 풀면 기존 가중치를 보존하면서 fine-tune 할 수 있습니다.
- **Score mode**: cls 로짓이 불안할 땐 `--score-mode obj`로 완화하고, 안정 후 `obj*cls`로 복원합니다. 다중 클래스라 하더라도 이 과정을 거치면 초반 폭락을 피할 수 있습니다.
- **DSI**: teacher와 분포가 다르면 오히려 val 지표가 떨어질 수 있으므로 단계적으로 켜고, `--dsi-weight-min/max`와 `--dsi-gam-*`로 비중을 모니터링하세요.
- **TBPTT**: `--tbptt-detach`를 켜면 타임스텝마다 `.detach()`를 수행해 메모리/안정성을 확보합니다. 끄고 싶다면 시퀀스 전체 loss를 누적한 뒤 한 번만 `backward()`하도록 코드를 수정해야 동일한 오류(“backward twice”)를 피할 수 있습니다.
- **멀티 클래스 Warm-start**: 1클래스 가중치를 불러와 `--num-classes>1`로 학습하면 cls 헤드 가중치만 새로 초기화됩니다. 이때 `score-mode obj`로 시작하고, epoch별로 cls 손실이 충분히 내려갔는지 로그(`cls=` 항목)를 확인하세요.

## 4. 자주 묻는 질문

| 질문 | 답변 |
| --- | --- |
| **train/val을 같은 데이터로 썼는데 val mAP가 떨어지는 이유?** | 손실 함수(연속)와 검증 파이프라인(NMS + score threshold)이 다르기 때문입니다. 특히 작은 배치/높은 학습률/랜덤 초기화된 cls 헤드가 겹치면 같은 샘플에서도 mAP가 크게 출렁입니다. |
| **증강/DSI를 켰더니 val이 급락한다** | train에는 증강/DSI가 들어가지만 val에는 들어가지 않아 분포 차이가 생깁니다. Stage 1에서 끈 상태로 안정화한 뒤, 큰 학습률 감소와 함께 천천히 켜는 것이 안전합니다. |
| **`Trying to backward through the graph a second time` 오류** | `--tbptt-detach`를 끄고 타임스텝마다 `.backward()`를 호출하면 LSTM 그래프를 재사용하려 하면서 발생합니다. 옵션을 켜거나, loss를 누적한 뒤 한 번만 `backward()` 하세요. |
| **DSI 손실이 뭘 하는가?** | student deep feature를 teacher(ViT) feature에 정렬하는 보조 손실입니다. 검증/추론에는 사용되지 않으므로, 반드시 mAP 향상과 1:1로 대응하지는 않습니다. |

이 가이드를 따라 Stage 1 → Stage 2 순서로 학습을 진행하면, 소량 데이터 검증부터 대량 데이터 학습까지 일관된 절차로 LSTM 모델을 안정적으로 훈련할 수 있습니다.
