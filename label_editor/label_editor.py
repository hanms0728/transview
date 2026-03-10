#!/usr/bin/env python3
"""Simple GUI label editor for reviewing the auto-label outputs.

이 툴은 auto_labels/batch_root_inference.py 로 생성된 이미지/라벨 폴더를
간단히 훑으면서 잘못 인식된 객체는 삭제하고, 누락된 객체는 추가할 수 있게
해준다. Matplotlib 인터랙션을 사용하므로 별도의 GUI 프레임워크가 필요 없다.

Usage
-----
python auto_labels/label_editor.py \
    --root ./root_dataset \
    --img-exts .jpg,.png \
    --image-dirs images,images_gt,. \
    --start-index 0

Controls
--------
* `n` / `right`  : 다음 이미지 (현재 라벨 자동 저장)
* `p` / `left`   : 이전 이미지
* `a`            : 추가 모드 토글 (센터, 첫 번째 포인트, 두 번째 포인트 순으로 3번 클릭)
* `esc`          : 추가 모드 취소
* `delete/backspace` : 선택된 라벨 삭제
* `f`            : 선택한 라벨의 앞/뒤 꼭짓점 뒤집기
* `s`            : 수동 저장
* `q`            : 프로그램 종료 (종료 시에도 변경 사항 자동 저장)
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon, Rectangle


# Use Korean-capable fonts if available and keep minus signs readable.
plt.rcParams["font.family"] = [
    "AppleGothic",
    "Apple SD Gothic Neo",
    "NanumGothic",
    "Malgun Gothic",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# Remove Matplotlib's default fullscreen toggle so "f" stays bound to label flipping.
fullscreen_keys = list(plt.rcParams.get("keymap.fullscreen", []))
if fullscreen_keys:
    plt.rcParams["keymap.fullscreen"] = [
        key for key in fullscreen_keys if key.lower() not in {"f", "ctrl+f"}
    ]


CLASS_COLOR_PALETTE = (
    "#ff1744",  # bright red
    "#00e676",  # vivid green
    "#2979ff",  # bold blue
    "#ff9100",
    "#9c27b0",
    "#00bcd4",
    "#d500f9",
    "#aeea00",
    "#ff6d00",
    "#00bfa5",
)


def class_color(class_id: int) -> str:
    idx = abs(int(class_id)) % len(CLASS_COLOR_PALETTE)
    return CLASS_COLOR_PALETTE[idx]


def lighten_color(color: str, amount: float = 0.3) -> str:
    amount = max(0.0, min(1.0, amount))
    r, g, b = mcolors.to_rgb(color)
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return mcolors.to_hex((r, g, b))


def order_poly_ccw(poly4: np.ndarray) -> np.ndarray:
    """Return a CCW-ordered quadrilateral for stable rendering."""
    c = poly4.mean(axis=0)
    ang = np.arctan2(poly4[:, 1] - c[1], poly4[:, 0] - c[0])
    idx = np.argsort(ang)
    return poly4[idx]


def parallelogram_from_pred_triangle(tri_pred: np.ndarray) -> np.ndarray:
    """tri_pred: [cx,cy,f1x,f1y,f2x,f2y,(score?)] -> (4,2) float32(CCW)."""
    coords = np.asarray(tri_pred[:6], dtype=np.float32)
    cx, cy, x2, y2, x3, y3 = coords.tolist()
    x2m, y2m = 2 * cx - x2, 2 * cy - y2
    x3m, y3m = 2 * cx - x3, 2 * cy - y3
    poly = np.array(
        [[x2, y2], [x3, y3], [x2m, y2m], [x3m, y3m]], dtype=np.float32
    )
    return order_poly_ccw(poly)


@dataclass
class LabelEntry:
    class_id: int
    cx: float
    cy: float
    f1x: float
    f1y: float
    f2x: float
    f2y: float
    score: Optional[float] = None

    def to_line(self) -> str:
        base = (
            f"{self.class_id} "
            f"{self.cx:.2f} {self.cy:.2f} "
            f"{self.f1x:.2f} {self.f1y:.2f} "
            f"{self.f2x:.2f} {self.f2y:.2f}"
        )
        if self.score is not None:
            base += f" {self.score:.4f}"
        return base

    def to_triangle(self) -> np.ndarray:
        return np.array(
            [self.cx, self.cy, self.f1x, self.f1y, self.f2x, self.f2y],
            dtype=np.float32,
        )

    def center_point(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def front_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.f1x, self.f1y), (self.f2x, self.f2y)

    def rear_points(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        c0x, c0y = self.center_point()
        return (
            (2 * c0x - self.f1x, 2 * c0y - self.f1y),
            (2 * c0x - self.f2x, 2 * c0y - self.f2y),
        )

    def flip_front_back(self) -> None:
        (r1x, r1y), (r2x, r2y) = self.rear_points()
        self.f1x, self.f1y = r1x, r1y
        self.f2x, self.f2y = r2x, r2y

    def clone(self) -> "LabelEntry":
        return LabelEntry(
            class_id=self.class_id,
            cx=self.cx,
            cy=self.cy,
            f1x=self.f1x,
            f1y=self.f1y,
            f2x=self.f2x,
            f2y=self.f2y,
            score=self.score,
        )


@dataclass
class LabelSample:
    label_path: str
    image_path: str
    dataset_dir: str


def parse_exts(exts: str) -> Tuple[str, ...]:
    cleaned = []
    for ext in (exts or "").split(","):
        e = ext.strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        cleaned.append(e.lower())
    return tuple(cleaned) if cleaned else (".jpg", ".jpeg", ".png")


def parse_image_dirs(image_dirs: str) -> Tuple[str, ...]:
    cleaned = []
    for token in (image_dirs or "").split(","):
        t = token.strip()
        if not t:
            continue
        cleaned.append(t)
    if "." not in cleaned:
        cleaned.append(".")
    return tuple(cleaned)


def parse_class_choices(raw: str) -> Optional[List[int]]:
    cleaned: List[int] = []
    seen = set()
    for token in (raw or "").split(","):
        t = token.strip()
        if not t:
            continue
        try:
            value = int(t)
        except ValueError:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned or None


def load_labels(label_path: str) -> List[LabelEntry]:
    entries: List[LabelEntry] = []
    if not os.path.isfile(label_path):
        return entries
    with open(label_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:7]))
                score = float(parts[7]) if len(parts) >= 8 else None
            except ValueError:
                continue
            entries.append(
                LabelEntry(
                    class_id=class_id,
                    cx=coords[0],
                    cy=coords[1],
                    f1x=coords[2],
                    f1y=coords[3],
                    f2x=coords[4],
                    f2y=coords[5],
                    score=score,
                )
            )
    return entries


def save_labels(label_path: str, entries: Sequence[LabelEntry]) -> None:
    lines = [entry.to_line() for entry in entries]
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def find_image_for_label(
    dataset_dir: str,
    base_name: str,
    image_dirs: Sequence[str],
    img_exts: Sequence[str],
) -> Optional[str]:
    checked = []
    for rel_dir in image_dirs:
        if rel_dir in ("", "."):
            search_dir = dataset_dir
        else:
            search_dir = os.path.join(dataset_dir, rel_dir)
        for ext in img_exts:
            candidate = os.path.join(search_dir, base_name + ext)
            checked.append(candidate)
            if os.path.isfile(candidate):
                return candidate
    return None


def collect_label_samples(
    root_dir: str, img_exts: Sequence[str], image_dirs: Sequence[str]
) -> List[LabelSample]:
    root_dir = os.path.abspath(root_dir)
    label_dirs = set()
    direct = os.path.join(root_dir, "labels")
    if os.path.isdir(direct):
        label_dirs.add(direct)

    for dirpath, dirnames, _ in os.walk(root_dir):
        if os.path.basename(dirpath) == "labels":
            label_dirs.add(dirpath)
            dirnames[:] = []

    samples: List[LabelSample] = []
    for labels_dir in sorted(label_dirs):
        dataset_dir = os.path.dirname(labels_dir)
        for name in sorted(os.listdir(labels_dir)):
            if not name.lower().endswith(".txt"):
                continue
            label_path = os.path.join(labels_dir, name)
            base_name = os.path.splitext(name)[0]
            img_path = find_image_for_label(
                dataset_dir, base_name, image_dirs, img_exts
            )
            if img_path is None:
                print(f"[WARN] 이미지 파일을 찾을 수 없습니다: {label_path}")
                continue
            samples.append(
                LabelSample(
                    label_path=label_path,
                    image_path=img_path,
                    dataset_dir=dataset_dir,
                )
            )
    return samples


class LabelEditorApp:
    def __init__(
        self,
        samples: Sequence[LabelSample],
        root_dir: str,
        start_index: int = 0,
        default_class: int = 0,
        class_choices: Optional[Sequence[int]] = None,
    ) -> None:
        if not samples:
            raise ValueError("라벨 파일을 찾을 수 없습니다.")
        self.samples = list(samples)
        self.root_dir = os.path.abspath(root_dir)
        self.idx = max(0, min(start_index, len(self.samples) - 1))
        self.default_class = default_class
        if class_choices:
            unique: List[int] = []
            seen: set[int] = set()
            for c in class_choices:
                ic = int(c)
                if ic in seen:
                    continue
                seen.add(ic)
                unique.append(ic)
            self.class_choices = unique or None
        else:
            self.class_choices = None
        self.current_class = self.default_class
        if self.class_choices and self.current_class not in self.class_choices:
            self.current_class = self.class_choices[0]

        self.entries: List[LabelEntry] = []
        self.selected_idx: Optional[int] = None
        self.mode = "idle"
        self.add_points: List[Tuple[float, float]] = []
        self.dirty = False
        self.drag_state: Optional[dict] = None
        self.copy_mark_indices: Set[int] = set()
        self.pending_copies: List[LabelEntry] = []
        self.roi_mode = False
        self.roi_points: List[Tuple[float, float]] = []
        self.delete_roi_rect: Optional[Tuple[float, float, float, float]] = None
        self.roi_patch: Optional[Rectangle] = None
        self.undo_stack: List[dict] = []
        self.max_undo = 50

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.axis("off")
        self.fig.subplots_adjust(bottom=0.11, top=0.93)
        self.image_artist = None
        self.polygon_patches: List[Polygon] = []
        self.point_markers: List = []
        self.add_markers: List = []

        self.help_text = self.fig.text(
            0.5,
            0.015,
            "n/p(or ←/→ or v/c): prev/next · a: add · esc: cancel add · d: delete · f: flip dir · drag point: move · r: toggle delete ROI · y: toggle copy · u: clear copy · ctrl+z: undo · s: save · 0-9: set class (selected/new) · q: quit",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#dddddd",
            bbox=dict(facecolor="black", alpha=0.4, pad=4),
        )
        self.status_text = self.fig.text(
            0.01,
            0.015,
            "",
            ha="left",
            va="bottom",
            fontsize=10,
            color="#ffeb3b",
            bbox=dict(facecolor="black", alpha=0.5, pad=4),
        )
        self.info_text = self.fig.text(
            0.01,
            0.985,
            "",
            ha="left",
            va="top",
            fontsize=10,
            color="#ffffff",
            bbox=dict(facecolor="black", alpha=0.4, pad=4),
        )
        self.labels_text = self.fig.text(
            0.99,
            0.93,
            "",
            ha="right",
            va="top",
            fontsize=9,
            color="#d1ffff",
            fontfamily="monospace",
            bbox=dict(facecolor="black", alpha=0.35, pad=6),
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def run(self) -> None:
        self.goto(self.idx, force=True)
        plt.show()

    def on_close(self, _event) -> None:
        if self.dirty:
            try:
                save_labels(self.samples[self.idx].label_path, self.entries)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 라벨 저장 실패: {exc}")

    def set_status(self, text: str) -> None:
        self.status_text.set_text(text)
        self.fig.canvas.draw_idle()

    def update_info(self) -> None:
        sample = self.samples[self.idx]
        rel_label = os.path.relpath(sample.label_path, self.root_dir)
        dirty_flag = " *" if self.dirty else ""
        copy_text = (
            f" · copy:{len(self.copy_mark_indices)}" if self.copy_mark_indices else ""
        )
        roi_text = " · ROI" if self.delete_roi_rect else ""
        self.info_text.set_text(
            f"{self.idx + 1}/{len(self.samples)}{dirty_flag} · {rel_label} · "
            f"labels: {len(self.entries)} · new-class: {self.current_class}{copy_text}{roi_text}"
        )

    def refresh_patches(self) -> None:
        for patch in self.polygon_patches:
            patch.remove()
        self.polygon_patches = []
        for marker in self.point_markers:
            marker.remove()
        self.point_markers = []

        for idx, entry in enumerate(self.entries):
            poly = parallelogram_from_pred_triangle(entry.to_triangle())
            base_color = class_color(entry.class_id)
            edge_color = "#ffeb3b" if idx == self.selected_idx else base_color
            patch = Polygon(
                poly,
                closed=True,
                fill=False,
                linewidth=2.2 if idx == self.selected_idx else 1.4,
                edgecolor=edge_color,
            )
            self.ax.add_patch(patch)
            self.polygon_patches.append(patch)
            self.point_markers.extend(self._plot_entry_points(idx, entry))

        self.fig.canvas.draw_idle()
        self.update_info()
        self.update_label_text()
        self.update_roi_patch()

    def update_label_text(self) -> None:
        if not self.entries:
            self.labels_text.set_text("(라벨 없음)")
        else:
            lines = []
            for idx, entry in enumerate(self.entries, start=1):
                is_sel = (idx - 1) == self.selected_idx
                is_copy = (idx - 1) in self.copy_mark_indices
                sel_marker = ">" if is_sel else " "
                copy_marker = "*" if is_copy else " "
                lines.append(f"{sel_marker}{copy_marker}{idx:02d}: {entry.to_line()}")
            max_lines = 22
            if len(lines) > max_lines:
                head = max_lines // 2
                tail = max_lines - head - 1
                display = lines[:head] + ["..."] + lines[-tail:]
            else:
                display = lines
            self.labels_text.set_text("\n".join(display))

    def _plot_entry_points(self, idx: int, entry: LabelEntry) -> List:
        """Draw front/center keypoints to help orientation checking."""
        markers = []
        is_sel = idx == self.selected_idx
        base_color = class_color(entry.class_id)
        # 중심
        cx, cy = entry.center_point()
        (center_marker,) = self.ax.plot(
            [cx],
            [cy],
            marker="o",
            linestyle="None",
            markersize=9 if is_sel else 7,
            markerfacecolor="#ffee58" if is_sel else "#fff59d",
            markeredgecolor="#000000",
            markeredgewidth=1.1,
            alpha=0.9,
        )
        markers.append(center_marker)
        # 앞쪽 두 포인트
        front_color = lighten_color(base_color, 0.35 if is_sel else 0.55)
        for px, py in entry.front_points():
            (marker,) = self.ax.plot(
                [px],
                [py],
                marker="o",
                linestyle="None",
                markersize=8 if is_sel else 6,
                markerfacecolor=front_color,
                markeredgecolor=base_color,
                markeredgewidth=1.0,
                alpha=0.95,
            )
            markers.append(marker)
        return markers

    def draw_add_markers(self) -> None:
        for marker in self.add_markers:
            marker.remove()
        self.add_markers = []
        colors = ["#ffeb3b", "#00e5ff", "#ff80ab"]
        for idx, (x, y) in enumerate(self.add_points):
            (marker,) = self.ax.plot(
                [x],
                [y],
                marker="o",
                color=colors[idx % len(colors)],
                markersize=8,
            )
            self.add_markers.append(marker)
        self.fig.canvas.draw_idle()

    def goto(self, new_idx: int, force: bool = False) -> None:
        if not force and (new_idx < 0 or new_idx >= len(self.samples)):
            self.set_status("더 이상 이동할 수 없습니다.")
            return
        if self.dirty:
            try:
                save_labels(self.samples[self.idx].label_path, self.entries)
                self.set_status("변경 사항 자동 저장 완료.")
            except Exception as exc:  # noqa: BLE001
                self.set_status(f"라벨 저장 실패: {exc}")
        self.prepare_pending_copies()
        self.idx = max(0, min(new_idx, len(self.samples) - 1))
        self.selected_idx = None
        self.mode = "idle"
        self.add_points.clear()
        self.draw_add_markers()
        self.drag_state = None
        if self.roi_mode:
            self.roi_mode = False
            self.roi_points.clear()
        self.load_current_sample()

    def load_current_sample(self) -> None:
        self.undo_stack.clear()
        sample = self.samples[self.idx]
        img_bgr = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"이미지를 열 수 없습니다: {sample.image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(img_rgb)
        else:
            self.image_artist.set_data(img_rgb)
        self.ax.set_title(os.path.relpath(sample.image_path, self.root_dir))
        self.entries = load_labels(sample.label_path)
        self.dirty = False
        copied = self.apply_pending_copies()
        removed = self.remove_labels_in_roi(update_view=False, announce=False)
        self.refresh_patches()
        if copied and removed:
            self.set_status(
                f"복사된 라벨을 추가하고 삭제 영역에서 {removed}개를 제거했습니다."
            )
        elif copied:
            self.set_status("복사된 라벨을 추가했습니다. 위치를 조정하세요.")
        elif removed:
            self.set_status(f"삭제 영역에서 {removed}개의 라벨을 제거했습니다.")
        else:
            self.set_status("이미지 로드 완료.")

    def pick_entry(self, x: float, y: float, max_dist: float = 35.0) -> Optional[int]:
        best_idx = None
        best_dist = max_dist ** 2
        for idx, entry in enumerate(self.entries):
            dist = (entry.cx - x) ** 2 + (entry.cy - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def pick_entry_point(
        self, x: float, y: float, max_dist: float = 20.0
    ) -> Optional[Tuple[int, str]]:
        best: Optional[Tuple[int, str]] = None
        best_dist = max_dist ** 2
        for idx, entry in enumerate(self.entries):
            candidates = [
                ("center", entry.center_point()),
                ("front1", entry.front_points()[0]),
                ("front2", entry.front_points()[1]),
            ]
            for name, (px, py) in candidates:
                dist = (px - x) ** 2 + (py - y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (idx, name)
        return best

    def start_drag(self, entry_idx: int, point: str, x: float, y: float) -> None:
        entry = self.entries[entry_idx]
        if point == "center":
            px, py = entry.center_point()
        elif point == "front1":
            (px, py) = entry.front_points()[0]
        else:
            (px, py) = entry.front_points()[1]
        offset = (px - x, py - y)
        self.drag_state = {
            "entry_idx": entry_idx,
            "point": point,
            "offset": offset,
            "start": {
                "cx": entry.cx,
                "cy": entry.cy,
                "f1x": entry.f1x,
                "f1y": entry.f1y,
                "f2x": entry.f2x,
                "f2y": entry.f2y,
            },
        }
        self.push_undo()
        self.selected_idx = entry_idx
        self.refresh_patches()
        self.set_status(
            f"{point} 포인트 드래그 중... (클릭을 떼면 완료됩니다.)"
        )

    def update_drag(self, x: float, y: float) -> None:
        if not self.drag_state:
            return
        entry = self.entries[self.drag_state["entry_idx"]]
        offset_x, offset_y = self.drag_state["offset"]
        px = x + offset_x
        py = y + offset_y
        start = self.drag_state["start"]
        point = self.drag_state["point"]
        if point == "center":
            entry.cx = px
            entry.cy = py
        elif point == "front1":
            entry.f1x = px
            entry.f1y = py
        else:
            entry.f2x = px
            entry.f2y = py
        self.dirty = True
        self.refresh_patches()

    def finish_drag(self) -> None:
        if not self.drag_state:
            return
        point = self.drag_state["point"]
        self.drag_state = None
        self.set_status(f"{point} 포인트 이동 완료.")

    def push_undo(self) -> None:
        snapshot = {
            "entries": [entry.clone() for entry in self.entries],
            "selected_idx": self.selected_idx,
            "copy_marks": set(self.copy_mark_indices),
            "dirty": self.dirty,
        }
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            self.set_status("실행 취소할 작업이 없습니다.")
            return
        snapshot = self.undo_stack.pop()
        self.entries = [entry.clone() for entry in snapshot["entries"]]
        self.selected_idx = snapshot["selected_idx"]
        self.copy_mark_indices = set(snapshot["copy_marks"])
        self.dirty = snapshot["dirty"]
        self.pending_copies = []
        self.refresh_patches()
        self.set_status("마지막 작업을 취소했습니다.")

    def update_roi_patch(self) -> None:
        if self.roi_patch is not None:
            self.roi_patch.remove()
            self.roi_patch = None
        if not self.delete_roi_rect:
            return
        x1, y1, x2, y2 = self.delete_roi_rect
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        patch = Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=1.8,
            edgecolor="#00ffc3",
            facecolor="none",
            linestyle="--",
        )
        self.ax.add_patch(patch)
        self.roi_patch = patch

    def toggle_copy_mark(self) -> None:
        if self.selected_idx is None:
            self.set_status("복사 표시할 라벨을 먼저 선택하세요.")
            return
        idx = self.selected_idx
        if idx in self.copy_mark_indices:
            self.copy_mark_indices.remove(idx)
            self.set_status(f"라벨 #{idx + 1} 복사 표시를 해제했습니다.")
        else:
            self.copy_mark_indices.add(idx)
            self.set_status(f"라벨 #{idx + 1}를 복사 대상에 추가했습니다.")
        self.update_info()
        self.update_label_text()

    def toggle_roi_mode(self) -> None:
        if self.roi_mode:
            self.roi_mode = False
            self.roi_points.clear()
            self.set_status("삭제 영역 지정을 취소했습니다.")
            return
        if self.delete_roi_rect:
            self.clear_roi()
            return
        if self.mode == "add":
            self.toggle_add_mode()
        self.roi_mode = True
        self.roi_points.clear()
        self.set_status("삭제 영역 지정: 첫 번째 코너를 클릭하세요.")

    def clear_roi(self) -> None:
        if not self.delete_roi_rect:
            self.set_status("활성화된 삭제 영역이 없습니다.")
            return
        self.delete_roi_rect = None
        self.update_roi_patch()
        self.set_status("삭제 영역을 해제했습니다.")
        self.update_info()

    def finalize_roi(self) -> None:
        if len(self.roi_points) < 2:
            return
        (x1, y1), (x2, y2) = self.roi_points[:2]
        if x1 == x2 or y1 == y2:
            self.set_status("삭제 영역이 너무 작습니다. 다시 지정하세요.")
            self.roi_points.clear()
            self.roi_mode = False
            return
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        self.delete_roi_rect = (xmin, ymin, xmax, ymax)
        self.roi_mode = False
        self.roi_points.clear()
        removed = self.remove_labels_in_roi(record_undo=True)
        self.update_roi_patch()
        if removed:
            self.set_status(f"삭제 영역 설정 완료. {removed}개의 라벨을 제거했습니다.")
        else:
            self.set_status("삭제 영역 설정 완료. 해당 영역에 라벨이 없습니다.")
        self.update_info()

    def clear_copy_marks(self) -> None:
        if not self.copy_mark_indices:
            self.set_status("복사 표시된 라벨이 없습니다.")
            return
        self.copy_mark_indices.clear()
        self.pending_copies = []
        self.update_info()
        self.update_label_text()
        self.set_status("모든 복사 표시를 해제했습니다.")

    def prepare_pending_copies(self) -> None:
        if not self.copy_mark_indices:
            self.pending_copies = []
            return
        clones = [
            self.entries[idx].clone()
            for idx in sorted(self.copy_mark_indices)
            if 0 <= idx < len(self.entries)
        ]
        self.pending_copies = clones

    def apply_pending_copies(self) -> bool:
        count = len(self.pending_copies)
        if count == 0:
            self.copy_mark_indices.clear()
            return False
        start_idx = len(self.entries)
        for entry in self.pending_copies:
            self.entries.append(entry.clone())
        new_indices = set(range(start_idx, start_idx + count))
        self.pending_copies = []
        self.copy_mark_indices = new_indices
        if new_indices:
            self.selected_idx = max(new_indices)
        self.dirty = True
        self.update_info()
        self.update_label_text()
        return True

    def remove_labels_in_roi(
        self,
        update_view: bool = True,
        announce: bool = False,
        record_undo: bool = False,
    ) -> int:
        if not self.delete_roi_rect:
            return 0
        x1, y1, x2, y2 = self.delete_roi_rect
        old_entries = self.entries
        new_entries: List[LabelEntry] = []
        new_marks: Set[int] = set()
        removed = 0
        new_selected: Optional[int] = None
        selected_idx = self.selected_idx
        for idx, entry in enumerate(old_entries):
            cx, cy = entry.center_point()
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                removed += 1
                continue
            new_entries.append(entry)
            if idx in self.copy_mark_indices:
                new_marks.add(len(new_entries) - 1)
            if selected_idx is not None and idx == selected_idx:
                new_selected = len(new_entries) - 1
        if removed:
            if record_undo:
                self.push_undo()
            self.entries = new_entries
            self.copy_mark_indices = new_marks
            self.selected_idx = new_selected
            self.dirty = True
            if update_view:
                self.refresh_patches()
            else:
                self.update_info()
                self.update_label_text()
            if announce:
                self.set_status(f"선택한 영역에서 {removed}개의 라벨을 삭제했습니다.")
        return removed

    def on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if self.roi_mode:
            self.roi_points.append((x, y))
            if len(self.roi_points) >= 2:
                self.finalize_roi()
            else:
                self.set_status("삭제 영역 지정: 두 번째 코너를 클릭하세요.")
            return

        if self.mode == "add":
            self.add_points.append((x, y))
            self.draw_add_markers()
            remaining = 3 - len(self.add_points)
            if remaining <= 0:
                self.finish_add()
            else:
                self.set_status(f"추가 모드: {remaining}개의 포인트를 더 찍어주세요.")
            return

        if event.button == 1:
            picked_point = self.pick_entry_point(x, y)
            if picked_point is not None:
                entry_idx, point_name = picked_point
                self.start_drag(entry_idx, point_name, x, y)
                return

        picked = self.pick_entry(x, y)
        if picked is None:
            self.selected_idx = None
            self.refresh_patches()
            self.set_status("선택된 라벨 없음.")
            return

        self.selected_idx = picked
        self.refresh_patches()
        entry = self.entries[picked]
        sc_text = (
            f"{entry.score:.3f}" if entry.score is not None else "없음"
        )
        self.set_status(f"라벨 #{picked + 1} 선택 (score={sc_text}).")

    def on_motion(self, event) -> None:
        if not self.drag_state or event.xdata is None or event.ydata is None:
            return
        if event.inaxes != self.ax:
            return
        self.update_drag(float(event.xdata), float(event.ydata))

    def on_release(self, event) -> None:
        if event.button != 1:
            return
        if self.drag_state is None:
            return
        self.finish_drag()

    def finish_add(self) -> None:
        if len(self.add_points) != 3:
            return
        self.push_undo()
        (cx, cy), (f1x, f1y), (f2x, f2y) = self.add_points
        new_entry = LabelEntry(
            class_id=self.current_class,
            cx=cx,
            cy=cy,
            f1x=f1x,
            f1y=f1y,
            f2x=f2x,
            f2y=f2y,
        )
        self.entries.append(new_entry)
        self.selected_idx = len(self.entries) - 1
        self.dirty = True
        self.mode = "idle"
        self.add_points.clear()
        self.draw_add_markers()
        self.refresh_patches()
        self.set_status(
            f"새 라벨이 추가되었습니다. (class={self.current_class})"
        )

    def delete_selected(self) -> None:
        if self.selected_idx is None:
            self.set_status("삭제할 라벨을 먼저 선택하세요.")
            return
        self.push_undo()
        removed = self.entries.pop(self.selected_idx)
        removed_idx = self.selected_idx
        self.selected_idx = None
        if removed_idx in self.copy_mark_indices:
            self.copy_mark_indices.remove(removed_idx)
        updated_marks: Set[int] = set()
        for idx in self.copy_mark_indices:
            if idx > removed_idx:
                updated_marks.add(idx - 1)
            elif idx < removed_idx:
                updated_marks.add(idx)
        self.copy_mark_indices = updated_marks
        self.dirty = True
        self.refresh_patches()
        self.set_status(
            f"라벨 삭제 완료 (cx={removed.cx:.1f}, cy={removed.cy:.1f})."
        )

    def flip_selected(self) -> None:
        if self.selected_idx is None:
            self.set_status("뒤집을 라벨을 먼저 선택하세요.")
            return
        self.push_undo()
        entry = self.entries[self.selected_idx]
        entry.flip_front_back()
        self.dirty = True
        self.refresh_patches()
        self.set_status("앞/뒤 꼭짓점을 뒤집었습니다.")

    def toggle_add_mode(self) -> None:
        if self.mode == "add":
            self.mode = "idle"
            self.add_points.clear()
            self.draw_add_markers()
            self.set_status("추가 모드 취소.")
        else:
            if self.drag_state is not None:
                self.drag_state = None
            self.mode = "add"
            self.add_points.clear()
            self.draw_add_markers()
            self.set_status(
                f"추가 모드: 센터/포인트1/포인트2 순으로 클릭하세요. (class={self.current_class})"
            )

    def set_current_class(self, new_class: int) -> None:
        if self.class_choices and new_class not in self.class_choices:
            allowed = ", ".join(map(str, self.class_choices))
            self.set_status(f"허용된 클래스는 [{allowed}] 입니다.")
            return
        new_class = int(new_class)
        if self.current_class == new_class:
            self.set_status(f"현재 클래스는 {self.current_class} 입니다.")
            return
        self.current_class = new_class
        self.update_info()
        self.set_status(f"새 라벨 클래스가 {self.current_class} 로 설정되었습니다.")

    def set_selected_class(self, new_class: int) -> None:
        if self.selected_idx is None:
            self.set_status("클래스를 변경할 라벨을 먼저 선택하세요.")
            return
        if self.class_choices and new_class not in self.class_choices:
            allowed = ", ".join(map(str, self.class_choices))
            self.set_status(f"허용된 클래스는 [{allowed}] 입니다.")
            return
        entry = self.entries[self.selected_idx]
        if entry.class_id == new_class:
            self.set_status(f"라벨 #{self.selected_idx + 1} 는 이미 클래스 {new_class} 입니다.")
            return
        self.push_undo()
        entry.class_id = int(new_class)
        self.dirty = True
        self.refresh_patches()
        self.set_status(
            f"라벨 #{self.selected_idx + 1} 의 클래스를 {entry.class_id} 로 변경했습니다."
        )

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("n", "right", "v"):
            self.goto(self.idx + 1)
        elif key in ("p", "left", "c"):
            self.goto(self.idx - 1)
        elif key in ("s", "ctrl+s"):
            try:
                save_labels(self.samples[self.idx].label_path, self.entries)
                self.dirty = False
                self.update_info()
                self.set_status("수동 저장 완료.")
            except Exception as exc:  # noqa: BLE001
                self.set_status(f"라벨 저장 실패: {exc}")
        elif key == "d":
            self.delete_selected()
        elif key == "f":
            self.flip_selected()
        elif key == "a":
            self.toggle_add_mode()
        elif key == "y":
            self.toggle_copy_mark()
        elif key == "u":
            self.clear_copy_marks()
        elif key == "r":
            self.toggle_roi_mode()
        elif key == "ctrl+z":
            self.undo()
        elif len(key) == 1 and key.isdigit():
            digit = int(key)
            if self.selected_idx is not None:
                self.set_selected_class(digit)
            else:
                self.set_current_class(digit)
        elif key == "escape":
            if self.mode == "add":
                self.toggle_add_mode()
        elif key == "q":
            if self.dirty:
                try:
                    save_labels(self.samples[self.idx].label_path, self.entries)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] 라벨 저장 실패: {exc}")
            plt.close(self.fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label 수정용 간단 GUI 도구"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="labels 디렉터리를 포함하는 루트 경로",
    )
    parser.add_argument(
        "--img-exts",
        type=str,
        default=".jpg,.jpeg,.png",
        help="이미지 확장자 리스트 (콤마 구분)",
    )
    parser.add_argument(
        "--image-dirs",
        type=str,
        default="images,images_gt,.",
        help="이미지 탐색 대상 디렉터리 (dataset 폴더 기준 상대경로, 콤마 구분)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="처음 열 라벨 파일의 인덱스",
    )
    parser.add_argument(
        "--default-class",
        type=int,
        default=0,
        help="새 라벨 추가 시 사용할 클래스 ID",
    )
    parser.add_argument(
        "--class-choices",
        type=str,
        default="",
        help="새 라벨 추가 시 순환할 클래스 ID 목록 (콤마 구분)",
    )
    args = parser.parse_args()

    img_exts = parse_exts(args.img_exts)
    image_dirs = parse_image_dirs(args.image_dirs)
    class_choices = parse_class_choices(args.class_choices)
    samples = collect_label_samples(args.root, img_exts, image_dirs)
    if not samples:
        raise SystemExit("[Error] 사용할 라벨 파일을 찾지 못했습니다.")

    app = LabelEditorApp(
        samples=samples,
        root_dir=args.root,
        start_index=args.start_index,
        default_class=args.default_class,
        class_choices=class_choices,
    )
    app.run()


if __name__ == "__main__":
    main()
