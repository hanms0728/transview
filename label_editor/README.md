# 라벨 편집기

Matplotlib 기반 2.5D BEV 라벨 수동 편집 도구

## 사용법

```bash
python label_editor/label_editor.py --root <데이터셋 경로>
```

<details>
<summary>예제</summary>

```bash
python label_editor/label_editor.py \
    --root ./dataset_example/dataset_example_carla_coshow_9
```

</details>

## 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--root` | labels 디렉터리를 포함하는 루트 경로 | `.` |
| `--start-index` | 처음 열 라벨 파일의 인덱스 | `0` |
| `--default-class` | 새 라벨 추가 시 클래스 ID | `0` |
| `--class-choices` | 순환할 클래스 ID 목록 (콤마 구분) | `""` |
| `--img-exts` | 이미지 확장자 (콤마 구분) | `.jpg,.jpeg,.png` |

## 조작법

### 프레임 이동
| 키 | 동작 |
|----|------|
| `N` / `→` / `V` | 다음 프레임 |
| `P` / `←` / `C` | 이전 프레임 |

### 라벨 편집
| 키 | 동작 |
|----|------|
| `A` | 라벨 추가 모드 (클릭 3점으로 삼각형 → 평행사변형 생성) |
| `D` | 선택한 라벨 삭제 |
| `F` | 선택한 라벨 좌우 반전 |
| `0`~`9` | 선택한 라벨의 클래스 변경 / 현재 클래스 설정 |
| `Ctrl+Z` | 실행 취소 |
| `ESC` | 추가 모드 해제 |

### 기타
| 키 | 동작 |
|----|------|
| `S` / `Ctrl+S` | 수동 저장 |
| `Y` | 복사 마크 토글 |
| `U` | 복사 마크 전체 해제 |
| `R` | ROI 모드 토글 |
| `Q` | 종료 (변경사항 자동 저장) |

### 마우스
- **좌클릭**: 라벨 선택 / 추가 모드에서 꼭짓점 찍기
- **드래그**: 선택한 라벨 이동
