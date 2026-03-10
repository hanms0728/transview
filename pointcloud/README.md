# 3D 시각화

Open3D 기반 포인트클라우드 + 차량 메쉬 오버레이 시각화 도구

## 사용법

```bash
python pointcloud/overlay_obj_on_ply.py \
    --global-ply <PLY 파일 경로> \
    --bev-label-dir <BEV 라벨 경로> \
    --vehicle-glb <차량 메쉬 경로>
```

<details>
<summary>예제</summary>

```bash
python pointcloud/overlay_obj_on_ply.py \
    --global-ply ./dataset_example/carla_pointcloud/carla_pointcloud.ply \
    --bev-label-dir ./results/inference/bev_labels \
    --vehicle-glb ./pointcloud/car.glb
```

</details>

## 조작법

| 키 | 동작 |
|----|------|
| `A` / `D` | 이전 / 다음 프레임 |
| `Space` | 재생 / 정지 |
| `Q` | 종료 |

## 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--fps` | 재생 FPS | `30.0` |
| `--play` | 자동 재생 시작 | `False` |
| `--force-legacy` | 레거시 Visualizer 사용 | `True` |
| `--estimate-z` | KDTree로 지면 높이 추정 | `False` |
| `--invert-bev-y` | BEV Y축 반전 | `True` |
| `--size-mode` | 차량 크기 모드 (`dynamic`, `fixed`) | `fixed` |
| `--height-scale` | Z축 높이 배율 | `1` |
