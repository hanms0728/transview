# inference.py
# YOLO11 2.5D temporal inference with ONNX (ConvLSTM/GRU hidden-state carry + seq reset)
# + BEV by per-pixel LUT (npz) with bilinear sampling (X,Y,Z) & robust masks

import os
import cv2
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
import onnxruntime as ort

# --- 유틸 임포트
from src.geometry_utils import parallelogram_from_triangle, tiny_filter_on_dets
from src.evaluation_utils import (
    decode_predictions,
    evaluate_single_image,
    compute_detection_metrics,
    compute_detection_metrics_per_class,
    orientation_error_deg,
    iou_polygon,
)

# --- Matplotlib (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MplPolygon = None
    Line2D = None
    _MATPLOTLIB_AVAILABLE = False


# --- 클래스별/필터링 유틸
def parse_class_conf_map(conf_str: Optional[str]) -> dict:
    """Parse '0:0.5,1:0.3' -> {0:0.5,1:0.3}"""
    if not conf_str:
        return {}
    mapping = {}
    for token in conf_str.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"잘못된 class-conf 형식: '{token}' (예: 0:0.5)")
        cid_str, thr_str = token.split(":", 1)
        mapping[int(cid_str.strip())] = float(thr_str.strip())
    return mapping


def parse_allowed_classes(cls_str: Optional[str]) -> Optional[set]:
    """Parse '0,1,2' -> {0,1,2}"""
    if not cls_str:
        return None
    allowed = set()
    for token in cls_str.split(","):
        token = token.strip()
        if token:
            allowed.add(int(token))
    return allowed if allowed else None


def filter_dets_by_class_and_conf(dets: List[dict], allowed_classes: Optional[set],
                                  class_conf_map: dict, default_conf: float) -> List[dict]:
    """클래스 허용목록 및 클래스별 confidence 임계값 적용"""
    filtered = []
    for d in dets:
        cls_id = int(d.get("cls", d.get("class_id", 0)))
        if allowed_classes is not None and cls_id not in allowed_classes:
            continue
        thr = class_conf_map.get(cls_id, default_conf)
        if float(d.get("score", 0.0)) < thr:
            continue
        filtered.append(d)
    return filtered


# --- I/O helpers (라벨/BEV/시각화)
_CLASS_COLOR_TABLE = [
    (0, 0, 255),    # red (BGR)
    (0, 165, 255),  # orange
    (255, 0, 0),    # blue
    (255, 0, 255),  # magenta
    (0, 215, 255),  # gold
]


def _class_color(cls_id: Optional[int]):
    if cls_id is None:
        return (0, 0, 255)
    idx = int(cls_id) % len(_CLASS_COLOR_TABLE)
    return _CLASS_COLOR_TABLE[idx]
def load_gt_triangles(label_path: str, return_cls: bool = False):
    if not os.path.isfile(label_path):
        empty_tris = np.zeros((0, 3, 2), dtype=np.float32)
        if return_cls:
            return empty_tris, np.zeros((0,), dtype=np.int64)
        return empty_tris
    tris = []
    cls_list = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 7:
                continue
            cls_id, p0x, p0y, p1x, p1y, p2x, p2y = p[:7]
            tris.append([[float(p0x), float(p0y)],
                         [float(p1x), float(p1y)],
                         [float(p2x), float(p2y)]])
            cls_list.append(int(float(cls_id)))
    if len(tris) == 0:
        empty_tris = np.zeros((0,3,2), dtype=np.float32)
        if return_cls:
            return empty_tris, np.zeros((0,), dtype=np.int64)
        return empty_tris
    tris_arr = np.asarray(tris, dtype=np.float32)
    if not return_cls:
        return tris_arr
    cls_arr = np.asarray(cls_list, dtype=np.int64) if cls_list else np.zeros((0,), dtype=np.int64)
    return tris_arr, cls_arr


def poly_from_tri(tri: np.ndarray) -> np.ndarray:
    p0, p1, p2 = tri[0], tri[1], tri[2]
    return parallelogram_from_triangle(p0, p1, p2).astype(np.float32)


# --- Homography BEV (txt/npy)
def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.copy()
    flat = pts.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([flat, ones], axis=1)
    proj = homog @ H.T
    denom = proj[:, 2:3]
    result = np.full_like(proj[:, :2], np.nan, dtype=np.float64)
    valid = np.abs(denom[:, 0]) > 1e-9
    if np.any(valid):
        result[valid] = proj[valid, :2] / denom[valid]
    return result.reshape(pts.shape)


def _read_h_matrix(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix.lower() == ".npy":
            data = np.load(path)
        else:
            data = np.loadtxt(path)
    except Exception:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 9:
        arr = arr.reshape(3, 3)
    if arr.shape != (3, 3):
        return None
    return arr


def load_homography(calib_dir: str, image_name: str, cache: dict, invert: bool = False) -> Optional[np.ndarray]:
    base = os.path.splitext(os.path.basename(image_name))[0]
    if base in cache:
        return cache[base]
    search_order = [base + ext for ext in (".txt", ".npy", ".csv")]
    H = None
    for candidate in search_order:
        c_path = Path(calib_dir) / candidate
        if c_path.is_file():
            H = _read_h_matrix(c_path)
            break
    if H is None:
        matches = sorted(Path(calib_dir).glob(base + ".*"))
        for c_path in matches:
            H = _read_h_matrix(c_path)
            if H is not None:
                break
    if H is not None and invert:
        try:
            H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H = None
    cache[base] = H
    return H


def compute_bev_properties_homography(tri):
    p0, p1, p2 = np.asarray(tri, dtype=np.float64)
    if not np.all(np.isfinite(tri)):
        return None
    front_center = (p1 + p2) / 2.0
    front_vec = front_center - p0
    yaw = math.degrees(math.atan2(front_vec[1], front_vec[0]))
    yaw = (yaw + 180) % 360 - 180
    poly = parallelogram_from_triangle(p0, p1, p2)
    edges = [np.linalg.norm(poly[(i+1) % 4] - poly[i]) for i in range(4)]
    length = max(edges)
    width = min(edges)
    center = poly.mean(axis=0)
    front_edge = (p1, p2)
    return (float(center[0]), float(center[1])), float(length), float(width), float(yaw), front_edge


def draw_pred_only(image_bgr, dets, save_path_img, save_path_txt, W, H, W0, H0):
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_txt), exist_ok=True)
    img = image_bgr.copy()
    sx, sy = float(W0) / float(W), float(H0) / float(H)

    lines = []
    tri_orig_list: List[np.ndarray] = []
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)
        score = float(d["score"])
        cls_id = int(d.get("class_id", d.get("cls", 0)))
        color = _class_color(cls_id)
        poly4 = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.int32)
        cv2.polylines(img, [poly4], isClosed=True, color=color, thickness=2)
        for pt in tri.astype(int):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 0), -1)
        cx, cy = int(tri[0][0]), int(tri[0][1])
        cv2.putText(img, f"{cls_id}:{score:.2f}", (cx, max(0, cy-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        tri_orig = tri.copy()
        tri_orig[:, 0] *= sx
        tri_orig[:, 1] *= sy
        tri_orig_list.append(tri_orig.copy())
        p0o, p1o, p2o = tri_orig[0], tri_orig[1], tri_orig[2]
        lines.append(
            f"{cls_id} {p0o[0]:.2f} {p0o[1]:.2f} {p1o[0]:.2f} {p1o[1]:.2f} "
            f"{p2o[0]:.2f} {p2o[1]:.2f} {score:.4f}"
        )

    cv2.imwrite(save_path_img, img)
    with open(save_path_txt, "w") as f:
        f.write("\n".join(lines))
    return tri_orig_list


def draw_pred_with_gt(image_bgr_resized, dets, gt_tris_resized, save_path_img_mix, iou_thr=0.5):
    os.makedirs(os.path.dirname(save_path_img_mix), exist_ok=True)
    img = image_bgr_resized.copy()
    for g in gt_tris_resized:
        poly_g = poly_from_tri(g).astype(np.int32)
        cv2.polylines(img, [poly_g], True, (0,255,0), 2)
        for k in range(3):
            x = int(round(float(g[k,0]))); y = int(round(float(g[k,1])))
            cv2.circle(img, (x, y), 3, (0,255,0), -1)
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)
        score = float(d["score"])
        cls_id = int(d.get("class_id", d.get("cls", 0)))
        color = _class_color(cls_id)
        poly4 = poly_from_tri(tri).astype(np.int32)
        cv2.polylines(img, [poly4], True, color, 2)
        for pt in tri.astype(int):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 0), -1)
        p0 = tri[0].astype(int)
        cv2.putText(img, f"{cls_id}:{score:.2f}", (int(p0[0]), max(0, int(p0[1])-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if gt_tris_resized.shape[0] > 0 and len(dets) > 0:
        det_idx_sorted = np.argsort([-float(d["score"]) for d in dets])
        matched = np.zeros((gt_tris_resized.shape[0],), dtype=bool)
        for di in det_idx_sorted:
            d = dets[di]
            tri_d = np.asarray(d["tri"], dtype=np.float32)
            poly_d = poly_from_tri(tri_d)
            best_j, best_iou = -1, 0.0
            for j, gtri in enumerate(gt_tris_resized):
                if matched[j]:
                    continue
                poly_g = poly_from_tri(gtri)
                iou = iou_polygon(poly_d, poly_g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                matched[best_j] = True
                p0 = tri_d[0].astype(int)
                cls_id = int(d.get("class_id", d.get("cls", 0)))
                color = _class_color(cls_id)
                cv2.putText(img, f"IoU {best_iou:.2f}",
                            (int(p0[0]), int(p0[1]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(save_path_img_mix, img)


def normalize_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


# --- BEV 시각화 (2D 이미지)
def _prepare_bev_canvas(polygons: List[np.ndarray], padding: float = 1.0, target: float = 800.0):
    if not polygons:
        return None
    pts = np.concatenate(polygons, axis=0)
    if pts.size == 0:
        return None
    min_x = float(np.nanmin(pts[:, 0]) - padding)
    max_x = float(np.nanmax(pts[:, 0]) + padding)
    min_y = float(np.nanmin(pts[:, 1]) - padding)
    max_y = float(np.nanmax(pts[:, 1]) + padding)
    range_x = max(max_x - min_x, 1e-3)
    range_y = max(max_y - min_y, 1e-3)
    scale = target / max(range_x, range_y)
    width = int(max(range_x * scale, 300))
    height = int(max(range_y * scale, 300))
    return {"min_x":min_x,"max_x":max_x,"min_y":min_y,"max_y":max_y,"scale":scale,"width":width,"height":height}


def _to_canvas(points: np.ndarray, params: dict) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xs = (pts[:, 0] - params["min_x"]) * params["scale"]
    ys = (params["max_y"] - pts[:, 1]) * params["scale"]
    return np.stack([xs, ys], axis=1).astype(np.int32)


def draw_bev_visualization(preds_bev: List[dict], gt_tris_bev: Optional[np.ndarray], save_path_img: str, title: str):
    pred_polys = [det["poly"] for det in preds_bev]
    gt_polys = [poly_from_tri(tri) for tri in gt_tris_bev] if gt_tris_bev is not None else []
    polygons = pred_polys + gt_polys
    params = _prepare_bev_canvas(polygons)
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)

    if params is None:
        canvas = np.full((360, 360, 3), 240, dtype=np.uint8)
        cv2.putText(canvas, "No BEV data", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(save_path_img, canvas)
        return

    if _MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=220)
            ax.set_facecolor("#f7f7f7")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontsize=11, pad=10)
            ax.set_xlabel("X (m)", fontsize=10)
            ax.set_ylabel("Y (m)", fontsize=10)
            ax.set_xlim(params["min_x"], params["max_x"])
            ax.set_ylim(params["max_y"], params["min_y"])
            if params["min_x"] <= 0.0 <= params["max_x"]:
                ax.axvline(0.0, color="#999999", linewidth=0.9, linestyle="--", alpha=0.8)
            if params["min_y"] <= 0.0 <= params["max_y"]:
                ax.axhline(0.0, color="#999999", linewidth=0.9, linestyle="--", alpha=0.8)
            legend_handles = []
            if gt_polys:
                for poly in gt_polys:
                    if not np.all(np.isfinite(poly)):
                        continue
                    patch = MplPolygon(poly, closed=True, fill=False, edgecolor="#27ae60", linewidth=1.8)
                    ax.add_patch(patch)
                    ax.scatter(poly[:, 0], poly[:, 1], s=8, color="#27ae60", alpha=0.9)
                legend_handles.append(Line2D([0], [0], color="#27ae60", lw=2, label="GT"))
            if preds_bev:
                dy = max(params["max_y"] - params["min_y"], 1e-3)
                text_offset = 0.015 * dy
                for det in preds_bev:
                    poly = det["poly"]
                    if not np.all(np.isfinite(poly)):
                        continue
                    patch = MplPolygon(poly, closed=True, fill=False, edgecolor="#e74c3c", linewidth=1.8)
                    ax.add_patch(patch)
                    center = det["center"]
                    ax.scatter(center[0], center[1], s=20, color="#e74c3c", alpha=0.9)
                    f1, f2 = det.get("front_edge", (None, None))
                    if f1 is not None and f2 is not None and np.all(np.isfinite(f1)) and np.all(np.isfinite(f2)):
                        ax.plot([f1[0], f2[0]], [f1[1], f2[1]], linewidth=2.2, color="#1f77b4")
                    label = f"{det['score']:.2f} / {det['yaw']:.1f}°"
                    ax.text(center[0], center[1] + text_offset, label, fontsize=7.5, color="#e74c3c",
                            ha="center", va="bottom",
                            bbox=dict(facecolor="#ffffff", alpha=0.6, edgecolor="none", pad=1.5))
                legend_handles.append(Line2D([0], [0], color="#e74c3c", lw=2, label="Pred"))
                legend_handles.append(Line2D([0], [0], color="#1f77b4", lw=3, label="Front edge"))
            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.75, fontsize=8)
            fig.tight_layout(pad=0.6)
            fig.savefig(save_path_img, dpi=220)
            plt.close(fig)
            return
        except Exception:
            plt.close("all")

    # OpenCV fallback
    canvas = np.full((params["height"], params["width"], 3), 255, dtype=np.uint8)
    axis_color = (120, 120, 120)
    axis_thickness = 1
    if params["min_x"] <= 0.0 <= params["max_x"]:
        axis_x = np.array([[0.0, params["min_y"]], [0.0, params["max_y"]]], dtype=np.float64)
        axis_x_px = _to_canvas(axis_x, params)
        cv2.line(canvas, tuple(axis_x_px[0]), tuple(axis_x_px[1]), axis_color, axis_thickness, cv2.LINE_AA)
    if params["min_y"] <= 0.0 <= params["max_y"]:
        axis_y = np.array([[params["min_x"], 0.0], [params["max_x"], 0.0]], dtype=np.float64)
        axis_y_px = _to_canvas(axis_y, params)
        cv2.line(canvas, tuple(axis_y_px[0]), tuple(axis_y_px[1]), axis_color, axis_thickness, cv2.LINE_AA)

    for det in preds_bev:
        poly = det["poly"]
        if not np.all(np.isfinite(poly)):
            continue
        poly_px = _to_canvas(poly, params)
        cv2.polylines(canvas, [poly_px], True, (0, 0, 255), 2)
        center_px = _to_canvas(np.asarray(det["center"]).reshape(1, 2), params)[0]
        cv2.circle(canvas, tuple(center_px), 4, (0, 0, 255), -1)

    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    coord_text = f"x:[{params['min_x']:.2f},{params['max_x']:.2f}]  y:[{params['min_y']:.2f},{params['max_y']:.2f}]"
    cv2.putText(canvas, coord_text, (10, params["height"] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.imwrite(save_path_img, canvas)


def compute_bev_properties(tri_xy, tri_z=None,
                           pitch_clamp_deg: float = 15.0,
                           use_roll: bool = False,
                           roll_threshold_deg: float = 2.0,
                           roll_clamp_deg: float = 8.0):
    """tri_xy (3,2), tri_z (3,) optional -> center, length, width, yaw, front_edge, cz, pitch, roll"""
    p0, p1, p2 = np.asarray(tri_xy, dtype=np.float64)
    if not np.all(np.isfinite(tri_xy)):
        return None

    # yaw from XY
    front_center = (p1 + p2) / 2.0
    front_vec = front_center - p0
    yaw = math.degrees(math.atan2(front_vec[1], front_vec[0]))
    yaw = (yaw + 180) % 360 - 180

    poly = parallelogram_from_triangle(p0, p1, p2)
    edges = [np.linalg.norm(poly[(i+1)%4]-poly[i]) for i in range(4)]
    length = max(edges)
    width = min(edges)
    center = poly.mean(axis=0)
    front_edge = (p1, p2)

    cz = 0.0
    pitch_deg = 0.0
    roll_deg = 0.0
    if tri_z is not None and np.all(np.isfinite(tri_z)):
        z0, z1, z2 = float(tri_z[0]), float(tri_z[1]), float(tri_z[2])
        cz = (z0 + z1 + z2) / 3.0

        # pitch: front vs rear along length
        z_front = 0.5 * (z1 + z2)
        dz_len = z_front - z0
        if length > 1e-6:
            pitch_rad = math.atan2(dz_len, length)
            pitch_deg = math.degrees(pitch_rad)
            pitch_deg = float(np.clip(pitch_deg, -pitch_clamp_deg, pitch_clamp_deg))

        # roll: left-right difference at front edge (optional)
        if use_roll:
            dz_lr = z2 - z1  # right - left
            if width > 1e-6:
                roll_rad = math.atan2(dz_lr, width)
                roll_deg = math.degrees(roll_rad)
                if abs(roll_deg) < roll_threshold_deg:
                    roll_deg = 0.0
                roll_deg = float(np.clip(roll_deg, -roll_clamp_deg, roll_clamp_deg))

    return (float(center[0]), float(center[1])), float(length), float(width), float(yaw), front_edge, float(cz), float(pitch_deg), float(roll_deg)

def compute_bev_properties_3d(
    tri_xy: np.ndarray,
    tri_z: np.ndarray,
    pitch_clamp_deg: float = 15.0,
    use_roll: bool = False,
    roll_threshold_deg: float = 2.0,
    roll_clamp_deg: float = 8.0,
    xy_scale: float = 1.0,
    z_scale: float = 1.0
):
    """3D 평면에서 실제 공간상의 length/width 계산. xy_scale로 BEV 스케일 보정."""
    if tri_xy is None or tri_z is None:
        return None
    tri_xy = np.asarray(tri_xy, dtype=np.float64)
    tri_z  = np.asarray(tri_z,  dtype=np.float64).reshape(-1)
    if tri_xy.shape != (3,2) or tri_z.shape != (3,) or not (np.all(np.isfinite(tri_xy)) and np.all(np.isfinite(tri_z))):
        return None

    # 물리 계산 전 스케일 보정
    sxy = float(xy_scale) if xy_scale is not None else 1.0
    sz  = float(z_scale)  if z_scale  is not None else 1.0
    if sxy <= 0: sxy = 1.0
    if sz  <= 0: sz  = 1.0

    tri_xy_unscaled = tri_xy / sxy
    tri_z_scaled    = tri_z * sz

    # 3D 포인트: P0(rear), P1(front-left), P2(front-right)
    P0 = np.array([tri_xy_unscaled[0,0], tri_xy_unscaled[0,1], tri_z_scaled[0]], dtype=np.float64)
    P1 = np.array([tri_xy_unscaled[1,0], tri_xy_unscaled[1,1], tri_z_scaled[1]], dtype=np.float64)
    P2 = np.array([tri_xy_unscaled[2,0], tri_xy_unscaled[2,1], tri_z_scaled[2]], dtype=np.float64)
    Pf = 0.5 * (P1 + P2)

    # 평면 법선
    n = np.cross(P1 - P0, P2 - P0)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return None
    n /= n_norm

    # 길이/폭 축
    L_hat = Pf - P0
    Ln = np.linalg.norm(L_hat)
    if Ln < 1e-9:
        return None
    L_hat /= Ln
    W_hat = np.cross(n, L_hat)
    Wn = np.linalg.norm(W_hat)
    if Wn < 1e-9:
        return None
    W_hat /= Wn

    # 실제 길이/폭
    length_m = abs(np.dot(Pf - P0, L_hat))
    width_m  = abs(np.dot(P2 - P1, W_hat))

    # 4번째 코너/센터
    P3 = P1 + (P2 - P0)
    center_3d = 0.25 * (P0 + P1 + P2 + P3)
    cz = float(center_3d[2])

    # yaw
    yaw = math.degrees(math.atan2(L_hat[1], L_hat[0]))
    yaw = (yaw + 180) % 360 - 180

    # pitch
    v = Pf - P0
    v = Pf - P0
    horiz_len = np.linalg.norm([v[0], v[1]])
    pitch_rad = math.atan2(v[2], max(horiz_len, 1e-9))
    pitch_deg = float(np.clip(math.degrees(pitch_rad), -pitch_clamp_deg, pitch_clamp_deg))

    # roll
    roll_deg = 0.0
    if use_roll and width_m > 1e-6:
        dz_lr = (P2[2] - P1[2])
        roll_rad = math.atan2(dz_lr, width_m)
        roll_deg = math.degrees(roll_rad)
        if abs(roll_deg) < roll_threshold_deg:
            roll_deg = 0.0
        roll_deg = float(np.clip(roll_deg, -roll_clamp_deg, roll_clamp_deg))

    # 2D 시각화용 폴리곤
    p0 = tri_xy[0]; p1 = tri_xy[1]; p2 = tri_xy[2]
    poly_xy = parallelogram_from_triangle(p0, p1, p2).astype(np.float32)
    front_edge = (tri_xy[1], tri_xy[2])
    center_xy = poly_xy.mean(axis=0)

    # 실제 단위 길이/폭 반환
    return (float(center_xy[0]), float(center_xy[1])), float(length_m), float(width_m), float(yaw), front_edge, float(cz), float(pitch_deg), float(roll_deg)

def write_bev_labels(save_path: str, bev_dets: List[dict], write_3d: bool = True):
    """write_3d=True면 cz/pitch/roll 포함, False면 legacy 포맷"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    for det in bev_dets:
        cls_id = int(det.get("class_id", det.get("cls", 0)))
        cx, cy = det["center"]
        length = det["length"]
        width = det["width"]
        yaw = det["yaw"]
        if write_3d:
            cz = det.get("cz", 0.0)
            pitch = det.get("pitch", 0.0)
            roll = det.get("roll", 0.0)
            lines.append(
                f"{cls_id} {cx:.4f} {cy:.4f} {cz:.4f} {length:.4f} {width:.4f} "
                f"{yaw:.2f} {pitch:.2f} {roll:.2f}"
            )
        else:
            lines.append(f"{cls_id} {cx:.4f} {cy:.4f} {length:.4f} {width:.4f} {yaw:.2f}")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def evaluate_single_image_bev(preds_bev: List[dict], gt_tris_bev: np.ndarray, iou_thr=0.5):
    gt_arr = np.asarray(gt_tris_bev)
    num_gt = gt_arr.shape[0] if gt_arr.ndim == 3 else 0
    preds_sorted = sorted(preds_bev, key=lambda d: d["score"], reverse=True)
    if num_gt == 0:
        records = [(det["score"], 0, 0.0, None) for det in preds_sorted]
        return records, 0
    gt_polys = [poly_from_tri(tri) for tri in gt_arr]
    matched = np.zeros((num_gt,), dtype=bool)
    records = []
    for det in preds_sorted:
        poly_d = det["poly"]
        if not np.all(np.isfinite(poly_d)):
            records.append((det["score"], 0, 0.0, None))
            continue
        best_iou, best_idx = 0.0, -1
        for idx, poly_g in enumerate(gt_polys):
            if matched[idx] or not np.all(np.isfinite(poly_g)):
                continue
            iou = iou_polygon(poly_d, poly_g)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            orient_err = orientation_error_deg(det["tri"], gt_arr[best_idx])
            records.append((det["score"], 1, best_iou, orient_err))
        else:
            records.append((det["score"], 0, best_iou, None))
    return records, int(matched.sum())


# --- LUT 기반 보간
def _lut_pick_valid_mask(lut):
    """LUT에서 valid mask 선택 (floor_mask > ground_valid_mask > valid_mask > fallback)"""
    X = np.asarray(lut["X"])
    Y = np.asarray(lut["Y"])
    H, W = X.shape

    for key in ("floor_mask", "ground_valid_mask", "valid_mask"):
        if key in lut:
            V = np.asarray(lut[key]).astype(bool)
            break
    else:
        V = np.isfinite(X) & np.isfinite(Y)

    if V.ndim == 1 and V.size == H * W:
        V = V.reshape(H, W)
    elif V.shape != (H, W):
        V = np.resize(V.astype(bool), (H, W))

    return V


def _bilinear_lut_xy(lut, u, v, min_valid_corners: int = 3, boundary_eps: float = 1e-3):
    """LUT에서 (u,v) 픽셀좌표를 bilinear 보간하여 (Xw, Yw, valid) 반환. min_valid_corners개 이상 유효 코너면 허용."""
    X = np.asarray(lut["X"])
    Y = np.asarray(lut["Y"])
    V = _lut_pick_valid_mask(lut)

    H, W = X.shape
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()

    # 경계 클램프
    eps = float(boundary_eps)
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-3
    u = np.clip(u, 0.0, W - 1 - eps)
    v = np.clip(v, 0.0, H - 1 - eps)

    Xw = np.full(u.shape, np.nan, dtype=np.float32)
    Yw = np.full(v.shape, np.nan, dtype=np.float32)
    valid = np.zeros(u.shape, dtype=bool)

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    ok = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)
    if not np.any(ok):
        return Xw, Yw, valid

    u0_ok = u0[ok]; v0_ok = v0[ok]
    u1_ok = u1[ok]; v1_ok = v1[ok]
    du = u[ok] - u0_ok
    dv = v[ok] - v0_ok

    X00 = X[v0_ok, u0_ok]; X10 = X[v0_ok, u1_ok]
    X01 = X[v1_ok, u0_ok]; X11 = X[v1_ok, u1_ok]
    Y00 = Y[v0_ok, u0_ok]; Y10 = Y[v0_ok, u1_ok]
    Y01 = Y[v1_ok, u0_ok]; Y11 = Y[v1_ok, u1_ok]
    V00 = V[v0_ok, u0_ok]; V10 = V[v0_ok, u1_ok]
    V01 = V[v1_ok, u0_ok]; V11 = V[v1_ok, u1_ok]

    # 유효 코너 개수 기준 통과
    min_c = int(min_valid_corners)
    min_c = 0 if min_c < 0 else (4 if min_c > 4 else min_c)
    nvalid = V00.astype(int) + V10.astype(int) + V01.astype(int) + V11.astype(int)
    allow = nvalid >= min_c

    if np.any(allow):
        w00 = (1.0 - du) * (1.0 - dv)
        w10 = du * (1.0 - dv)
        w01 = (1.0 - du) * dv
        w11 = du * dv

        # 유효 코너만 가중치 반영 + 재정규화
        w00[~V00] = 0.0; w10[~V10] = 0.0; w01[~V01] = 0.0; w11[~V11] = 0.0
        wsum = w00 + w10 + w01 + w11
        wsum[wsum == 0.0] = 1.0
        w00 /= wsum; w10 /= wsum; w01 /= wsum; w11 /= wsum

        Xw_ok = (w00 * X00 + w10 * X10 + w01 * X01 + w11 * X11).astype(np.float32)
        Yw_ok = (w00 * Y00 + w10 * Y10 + w01 * Y01 + w11 * Y11).astype(np.float32)

        whole = np.zeros_like(ok, dtype=bool)
        whole[ok] = allow
        Xw[whole] = Xw_ok[allow]
        Yw[whole] = Yw_ok[allow]
        valid[whole] = True

    return Xw, Yw, valid


def _bilinear_lut_xyz(lut, u, v, min_valid_corners: int = 3, boundary_eps: float = 1e-3):
    """XY 보간 + Z 확장. LUT에 Z 없으면 NaN."""
    Xw, Yw, valid = _bilinear_lut_xy(
        lut, u, v,
        min_valid_corners=min_valid_corners,
        boundary_eps=boundary_eps
    )
    Z = lut.get("Z", None)
    if Z is None:
        Zw = np.full_like(Xw, np.nan, dtype=np.float32)
        return Xw, Yw, Zw, valid

    Z = np.asarray(Z)
    H, W = Z.shape
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()

    eps = float(boundary_eps) if np.isfinite(boundary_eps) and boundary_eps > 0 else 1e-3
    u = np.clip(u, 0.0, W - 1 - eps)
    v = np.clip(v, 0.0, H - 1 - eps)

    Zw = np.full(u.shape, np.nan, dtype=np.float32)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1
    ok = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)

    if np.any(ok):
        u0_ok = u0[ok]; v0_ok = v0[ok]
        u1_ok = u1[ok]; v1_ok = v1[ok]
        du = u[ok] - u0_ok
        dv = v[ok] - v0_ok

        Z00 = Z[v0_ok, u0_ok]; Z10 = Z[v0_ok, u1_ok]
        Z01 = Z[v1_ok, u0_ok]; Z11 = Z[v1_ok, u1_ok]

        w00 = (1.0 - du) * (1.0 - dv)
        w10 = du * (1.0 - dv)
        w01 = (1.0 - du) * dv
        w11 = du * dv

        Zw_ok = (w00 * Z00 + w10 * Z10 + w01 * Z01 + w11 * Z11).astype(np.float32)
        Zw[ok] = Zw_ok
        Zw[~valid] = np.nan  # XY invalid → Z도 무효

    return Xw, Yw, Zw, valid


def tris_img_to_bev_by_lut(tris_img: np.ndarray, lut_data: dict, bev_scale: float = 1.0,
                            min_valid_corners: int = 3, boundary_eps: float = 1e-3):
    """이미지 좌표 삼각형 [N,3,2]을 LUT로 BEV 투영. -> (tris_bev_xy, tris_bev_z, tri_ok)"""
    if tris_img.size == 0:
        return (np.zeros((0,3,2), dtype=np.float32),
                np.zeros((0,3), dtype=np.float32),
                np.zeros((0,), dtype=bool))

    u = tris_img[:, :, 0].reshape(-1)
    v = tris_img[:, :, 1].reshape(-1)

    Xw, Yw, Zw, valid = _bilinear_lut_xyz(
        lut_data, u, v,
        min_valid_corners=min_valid_corners,
        boundary_eps=boundary_eps
    )

    # tri 단위 유효성: 각 tri의 3점 모두 valid
    valid = valid.reshape(-1, 3)
    tri_ok = np.all(valid, axis=1)

    Xw = Xw.reshape(-1, 3)
    Yw = Yw.reshape(-1, 3)
    Zw = Zw.reshape(-1, 3)

    tris_bev_xy = np.stack([Xw, Yw], axis=-1).astype(np.float32)  # [N,3,2]
    tris_bev_xy *= float(bev_scale)

    return tris_bev_xy, Zw.astype(np.float32), tri_ok.astype(bool)


# --- ONNX temporal runner
class ONNXTemporalRunner:
    """ConvLSTM/GRU ONNX 모델 실행기. hidden state carry + seq reset 지원."""
    def __init__(self, onnx_path, providers=("CUDAExecutionProvider","CPUExecutionProvider"),
                 state_stride_hint: int = 32, default_hidden_ch: int = 256):
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.outs = [o.name for o in self.sess.get_outputs()]

        # 입력/출력 이름 매핑
        cand_x = [n for n in self.inputs if n.lower() in ("images","image","input")]
        self.x_name = cand_x[0] if cand_x else list(self.inputs.keys())[0]
        self.h_name = next((n for n in self.inputs if "h_in" in n.lower()), None)
        self.c_name = next((n for n in self.inputs if "c_in" in n.lower()), None)

        self.ho_name = next((n for n in self.outs if "h_out" in n.lower()), None)
        self.co_name = next((n for n in self.outs if "c_out" in n.lower()), None)
        self.reg_names = [n for n in self.outs if "reg" in n.lower()]
        self.obj_names = [n for n in self.outs if "obj" in n.lower()]
        self.cls_names = [n for n in self.outs if "cls" in n.lower()]

        def _sort_key(s):
            toks = []
            acc = ""
            for ch in s:
                if ch.isdigit():
                    acc += ch
                else:
                    if acc:
                        toks.append(int(acc))
                        acc = ""
                    toks.append(ch)
            if acc:
                toks.append(int(acc))
            return tuple(toks)

        self.reg_names.sort(key=_sort_key)
        self.obj_names.sort(key=_sort_key)
        self.cls_names.sort(key=_sort_key)
        self.num_scales = min(len(self.reg_names), len(self.obj_names), len(self.cls_names))

        # hidden state 버퍼
        self.h_buf = None
        self.c_buf = None

        # shape 메타
        self.state_stride_hint = int(state_stride_hint)
        self.default_hidden_ch = int(default_hidden_ch)
        self.h_shape_meta = self._shape_from_input_meta(self.h_name)
        self.c_shape_meta = self._shape_from_input_meta(self.c_name)

    def _shape_from_input_meta(self, name):
        if name is None:
            return None
        meta = self.inputs[name].shape  # [N,C,Hs,Ws] with possible None
        def _to_int(val, default):
            return int(val) if isinstance(val, (int, np.integer)) else default
        N = _to_int(meta[0], 1)
        C = _to_int(meta[1], self.default_hidden_ch)
        Hs = _to_int(meta[2], 0)
        Ws = _to_int(meta[3], 0)
        return [N, C, Hs, Ws]

    def reset(self):
        self.h_buf = None
        self.c_buf = None

    def _ensure_state(self, img_numpy_chw: np.ndarray):
        _, _, H, W = img_numpy_chw.shape
        if self.h_name and self.h_buf is None:
            N, C, Hs, Ws = self.h_shape_meta
            if Hs == 0 or Ws == 0:
                Hs = max(1, H // self.state_stride_hint)
                Ws = max(1, W // self.state_stride_hint)
            self.h_buf = np.zeros((N, C, Hs, Ws), dtype=np.float32)
        if self.c_name and self.c_buf is None:
            N, C, Hs, Ws = self.c_shape_meta
            if Hs == 0 or Ws == 0:
                Hs = max(1, H // self.state_stride_hint)
                Ws = max(1, W // self.state_stride_hint)
            self.c_buf = np.zeros((N, C, Hs, Ws), dtype=np.float32)

    def forward(self, img_numpy_chw):
        """(1,3,H,W) float32 입력 -> list of (reg,obj,cls) torch.Tensor"""
        self._ensure_state(img_numpy_chw)

        feeds = {self.x_name: img_numpy_chw}
        if self.h_name is not None and self.h_buf is not None:
            feeds[self.h_name] = self.h_buf
        if self.c_name is not None and self.c_buf is not None:
            feeds[self.c_name] = self.c_buf

        outs = self.sess.run(self.outs, feeds)
        out_map = {n: v for n, v in zip(self.outs, outs)}

        # 상태 갱신
        if self.ho_name:
            self.h_buf = out_map[self.ho_name]
        if self.co_name:
            self.c_buf = out_map[self.co_name]

        # (reg, obj, cls) torch.Tensor 리스트로 변환
        pred_list = []
        for rn, on, cn in zip(self.reg_names, self.obj_names, self.cls_names):
            pr = torch.from_numpy(out_map[rn])
            po = torch.from_numpy(out_map[on])
            pc = torch.from_numpy(out_map[cn])
            pred_list.append((pr, po, pc))
        return pred_list


# --- 시퀀스 키
def seq_key(file_path: str, mode: str) -> str:
    p = Path(file_path)
    if mode == "by_subdir":
        return p.parent.name
    stem = p.stem
    if "_" in stem:
        return stem.split("_")[0]
    if "-" in stem:
        return stem.split("-")[0]
    return "ALL"

def _sane_dims(L, W, args) -> bool:
    if not (np.isfinite(L) and np.isfinite(W)):
        return False
    if not (args.min_length <= L <= args.max_length): return False
    if not (args.min_width  <= W <= args.max_width ): return False
    r = L / max(W, 1e-6)
    if not (args.min_lw_ratio <= r <= args.max_lw_ratio): return False
    return True


# --- 메인
def main():
    ap = argparse.ArgumentParser("YOLO11 2.5D ONNX Temporal Inference (+GT & BEV via LUT)")
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="./results/inference")
    ap.add_argument("--weights", type=str, required=True, help="ONNX 파일 경로")
    ap.add_argument("--img-size", type=str, default="864,1536")
    ap.add_argument("--score-mode", type=str, default="obj*cls", choices=["obj","cls","obj*cls"])
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms-iou", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--contain-thr", type=float, default=0.85)
    ap.add_argument("--clip-cells", type=float, default=None)
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png")
    ap.add_argument("--class-conf-map", type=str, default=None,
                    help="클래스별 confidence 임계값 (예: '0:0.7,1:0.5,2:0.3')")
    ap.add_argument("--allowed-classes", type=str, default=None,
                    help="허용할 클래스 ID 목록 (예: '0,1')")

    # strides for decoder
    ap.add_argument("--strides", type=str, default="8,16,32")

    # temporal & sequence
    ap.add_argument("--temporal", type=str, default="lstm", choices=["none","gru","lstm"])
    ap.add_argument("--seq-mode", type=str, default="by_prefix", choices=["by_prefix","by_subdir"])
    ap.add_argument("--reset-per-seq", action="store_true", default=True)

    # state feature map hint
    ap.add_argument("--state-stride-hint", type=int, default=32)
    ap.add_argument("--default-hidden-ch", type=int, default=256)

    # Eval(2D)
    ap.add_argument("--gt-label-dir", type=str, default=None)
    ap.add_argument("--eval-iou-thr", type=float, default=0.5)
    ap.add_argument("--labels-are-original-size", action="store_true", default=True)

    # BEV mode selection
    ap.add_argument("--bev-mode", type=str, default="homography", choices=["lut", "homography"],
                    help="BEV 변환 방식: lut (npz LUT) 또는 homography (txt/npy 호모그래피)")

    # BEV via LUT
    ap.add_argument(
        "--lut-path",
        type=str,
        default=None,
        help="pixel2world_lut.npz 경로 (bev-mode=lut 시 사용)",
    )

    # BEV via Homography
    ap.add_argument("--calib-dir", type=str, default=None,
                    help="호모그래피 행렬 디렉토리 (bev-mode=homography 시 사용)")
    ap.add_argument("--invert-calib", action="store_true",
                    help="호모그래피 행렬을 역행렬로 변환")

    ap.add_argument("--bev-scale", type=float, default=1.0)
    
    # LUT interpolation robustness
    ap.add_argument("--lut-min-corners", type=int, default=3,
                    help="Min number of valid bilinear corners (0..4) to accept a sample (default: 3)")
    ap.add_argument("--lut-boundary-eps", type=float, default=1e-3,
                    help="Clamp (u,v) to [0,W-1-eps]/[0,H-1-eps] (default: 1e-3)")

    # BEV 3D label options
    ap.add_argument("--bev-label-3d", action="store_true", default=True,
                    help="Write BEV labels with cz, yaw, pitch, roll (default: on)")
    ap.add_argument("--use-roll", action="store_true", default=False,
                    help="Also estimate & write roll from left-right Z difference")
    ap.add_argument("--roll-threshold-deg", type=float, default=2.0,
                    help="Absolute roll below this (deg) is snapped to 0")
    ap.add_argument("--roll-clamp-deg", type=float, default=8.0,
                    help="Clamp |roll| to this maximum (deg)")
    ap.add_argument("--pitch-clamp-deg", type=float, default=30.0,
                    help="Clamp |pitch| to this maximum (deg)")

    # 3D 크기 필터
    ap.add_argument("--min-length", type=float, default=0.0)
    ap.add_argument("--max-length", type=float, default=100.0)
    ap.add_argument("--min-width",  type=float, default=0.0)
    ap.add_argument("--max-width",  type=float, default=100.0)
    ap.add_argument("--min-lw-ratio", type=float, default=0.01)
    ap.add_argument("--max-lw-ratio", type=float, default=100)

    # onnxruntime providers
    ap.add_argument("--no-cuda", action="store_true", help="CUDA EP 비활성화")

    args = ap.parse_args()
    H, W = map(int, args.img_size.split(","))
    strides = [float(s) for s in args.strides.split(",")]

    class_conf_map = parse_class_conf_map(args.class_conf_map)
    allowed_classes = parse_allowed_classes(args.allowed_classes)
    if class_conf_map:
        print(f"[Filter] class-conf-map: {class_conf_map}")
    if allowed_classes:
        print(f"[Filter] allowed-classes: {allowed_classes}")

    providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    if args.no_cuda or ort.get_device().upper() != "GPU":
        providers = ["CPUExecutionProvider"]

    runner = ONNXTemporalRunner(
        args.weights, providers=providers,
        state_stride_hint=args.state_stride_hint,
        default_hidden_ch=args.default_hidden_ch
    )

    if len(strides) != runner.num_scales:
        if runner.num_scales == len(strides) + 1:
            # 추론 모델이 스케일을 하나 더 갖고 있음 (예: 4,8,16,32)
            first = max(1.0, strides[0] / 2.0)
            strides = [first] + strides
            print(f"[Infer-ONNX] strides length adjusted to match model outputs: {strides}")
        elif len(strides) > runner.num_scales:
            strides = strides[:runner.num_scales]
            print(f"[Infer-ONNX] strides truncated to {strides} to match model outputs")
        else:
            raise ValueError(f"ONNX outputs ({runner.num_scales}) and strides ({len(strides)}) mismatch. "
                             "Set --strides accordingly (e.g., 4,8,16,32 for stride-4 models).")

    out_img_dir = os.path.join(args.output_dir, "images")
    out_lab_dir = os.path.join(args.output_dir, "labels")
    out_mix_dir = os.path.join(args.output_dir, "images_with_gt")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)
    os.makedirs(out_mix_dir, exist_ok=True)

    # BEV 초기화
    lut_data = None
    homography_cache = {}
    missing_h_names = set()

    if args.bev_mode == "lut" and args.lut_path:
        use_bev = True
        if not os.path.isfile(args.lut_path):
            raise FileNotFoundError(f"LUT not found: {args.lut_path}")
        lut_data = dict(np.load(args.lut_path))
        print(f"[BEV] mode=lut, LUT: {args.lut_path}")
    elif args.bev_mode == "homography" and args.calib_dir:
        use_bev = True
        if not os.path.isdir(args.calib_dir):
            raise FileNotFoundError(f"Calib dir not found: {args.calib_dir}")
        print(f"[BEV] mode=homography, calib: {args.calib_dir}")
    else:
        use_bev = False
        print("[BEV] No BEV path provided; skipping BEV outputs.")

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    names = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
    names.sort()

    do_eval_2d = args.gt_label_dir is not None and os.path.isdir(args.gt_label_dir)

    metric_records = []
    total_gt = 0
    total_gt_by_class = {}

    # BEV 출력 디렉토리
    metric_records_bev = []
    total_gt_bev = 0
    total_gt_bev_by_class = {}
    out_bev_img_dir = None
    out_bev_mix_dir = None
    out_bev_lab_dir = None
    if use_bev:
        out_bev_img_dir = os.path.join(args.output_dir, "bev_images")
        out_bev_mix_dir = os.path.join(args.output_dir, "bev_images_with_gt")
        out_bev_lab_dir = os.path.join(args.output_dir, "bev_labels")
        os.makedirs(out_bev_img_dir, exist_ok=True)
        os.makedirs(out_bev_mix_dir, exist_ok=True)
        os.makedirs(out_bev_lab_dir, exist_ok=True)

    print(
        f"[Infer-ONNX] imgs={len(names)}, temporal={args.temporal}, seq={args.seq_mode}, "
        f"reset_per_seq={args.reset_per_seq}, eval2D={do_eval_2d}, use_bev={use_bev}"
    )

    prev_key = None
    for name in tqdm(names, desc="[Infer-ONNX]"):
        path = os.path.join(args.input_dir, name)

        # 시퀀스 경계 판단 → reset
        k = seq_key(path, args.seq_mode)
        if args.reset_per_seq and k != prev_key:
            runner.reset()
        prev_key = k

        img_bgr0 = cv2.imread(path)
        if img_bgr0 is None:
            continue

        H0, W0 = img_bgr0.shape[:2]
        img_bgr = cv2.resize(img_bgr0, (W, H), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_np = img_rgb.transpose(2,0,1).astype(np.float32) / 255.0
        img_np = np.expand_dims(img_np, 0)  # (1,3,H,W)

        scale_resize_x = W / float(W0)
        scale_resize_y = H / float(H0)
        scale_to_orig_x = float(W0) / float(W)
        scale_to_orig_y = float(H0) / float(H)

        outs = runner.forward(img_np)

        # decode
        dets = decode_predictions(
            outs, strides,
            clip_cells=args.clip_cells,
            conf_th=min(list(class_conf_map.values()) + [args.conf]) if class_conf_map else args.conf,
            nms_iou=args.nms_iou,
            topk=args.topk,
            contain_thr=args.contain_thr,
            score_mode=args.score_mode,
            use_gpu_nms=True
        )[0]

        dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)
        dets = filter_dets_by_class_and_conf(dets, allowed_classes, class_conf_map, args.conf)

        save_img = os.path.join(out_img_dir, name)
        save_txt = os.path.join(out_lab_dir, os.path.splitext(name)[0] + ".txt")
        pred_tris_orig = draw_pred_only(img_bgr, dets, save_img, save_txt, W, H, W0, H0)

        # 2D Eval
        gt_for_eval = np.zeros((0, 3, 2), dtype=np.float32)
        gt_tri_orig_for_bev = np.zeros((0, 3, 2), dtype=np.float32)
        gt_cls_eval = np.zeros((0,), dtype=np.int64)

        if do_eval_2d:
            lab_path = os.path.join(args.gt_label_dir, os.path.splitext(name)[0] + ".txt")
            gt_tri_raw, gt_cls_raw = load_gt_triangles(lab_path, return_cls=True)
            if gt_tri_raw.shape[0] > 0:
                if args.labels_are_original_size:
                    gt_tri_orig_for_bev = gt_tri_raw.astype(np.float32)
                    gt_for_eval = gt_tri_raw.copy()
                    gt_for_eval[:, :, 0] *= scale_resize_x
                    gt_for_eval[:, :, 1] *= scale_resize_y
                else:
                    gt_for_eval = gt_tri_raw.astype(np.float32)
                    gt_tri_orig_for_bev = gt_tri_raw.copy()
                    gt_tri_orig_for_bev[:, :, 0] *= scale_to_orig_x
                    gt_tri_orig_for_bev[:, :, 1] *= scale_to_orig_y
                gt_cls_eval = gt_cls_raw.astype(np.int64)

            save_img_mix = os.path.join(out_mix_dir, name)
            draw_pred_with_gt(img_bgr, dets, gt_for_eval, save_img_mix, iou_thr=args.eval_iou_thr)

            if gt_for_eval.shape[0] > 0:
                records, _ = evaluate_single_image(
                    dets, gt_for_eval, gt_cls_eval, iou_thr=args.eval_iou_thr
                )
                metric_records.extend(records)
                total_gt += gt_for_eval.shape[0]
                for c in gt_cls_eval:
                    total_gt_by_class[int(c)] = total_gt_by_class.get(int(c), 0) + 1

        # ---- BEV ----
        if use_bev:
            bev_dets = []

            if args.bev_mode == "lut":
                # LUT 방식
                if pred_tris_orig:
                    pred_stack_orig = np.asarray(pred_tris_orig, dtype=np.float64)
                    pred_tris_bev_xy, pred_tris_bev_z, good_mask = tris_img_to_bev_by_lut(
                        pred_stack_orig, lut_data, bev_scale=float(args.bev_scale),
                        min_valid_corners=int(args.lut_min_corners),
                        boundary_eps=float(args.lut_boundary_eps)
                    )
                    num_tri = min(len(dets), pred_tris_bev_xy.shape[0])
                    if len(dets) != pred_tris_bev_xy.shape[0]:
                        print(
                            f"[WARN] BEV LUT triangles ({pred_tris_bev_xy.shape[0]}) "
                            f"and detections ({len(dets)}) differ; clipping to {num_tri}."
                        )
                    for idx in range(num_tri):
                        det = dets[idx]
                        if not good_mask[idx]:
                            continue
                        tri_bev_xy = pred_tris_bev_xy[idx]
                        tri_bev_z = pred_tris_bev_z[idx]
                        if not np.all(np.isfinite(tri_bev_xy)):
                            continue
                        poly_bev = poly_from_tri(tri_bev_xy)
                        props = None
                        try:
                            props = compute_bev_properties_3d(
                                tri_bev_xy,
                                tri_bev_z,
                                pitch_clamp_deg=args.pitch_clamp_deg,
                                use_roll=args.use_roll,
                                roll_threshold_deg=args.roll_threshold_deg,
                                roll_clamp_deg=args.roll_clamp_deg,
                            )
                        except Exception:
                            props = None

                        if props is None:
                            props = compute_bev_properties(
                                tri_bev_xy,
                                tri_bev_z,
                                pitch_clamp_deg=args.pitch_clamp_deg,
                                use_roll=args.use_roll,
                                roll_threshold_deg=args.roll_threshold_deg,
                                roll_clamp_deg=args.roll_clamp_deg,
                            )
                        if props is None:
                            continue

                        center, length, width, yaw, front_edge, cz, pitch_deg, roll_deg = props

                        if not _sane_dims(length, width, args):
                            continue

                        bev_dets.append(
                            {
                                "score": float(det["score"]),
                                "tri": tri_bev_xy,
                                "poly": poly_bev,
                                "center": center,
                                "length": length,
                                "width": width,
                                "yaw": yaw,
                                "front_edge": front_edge,
                                "cz": cz,
                                "pitch": pitch_deg,
                                "roll": roll_deg if args.use_roll else 0.0,
                                "class_id": int(det.get("class_id", det.get("cls", 0))),
                            }
                        )

                # GT → BEV (LUT)
                if gt_tri_orig_for_bev.size > 0:
                    gt_u = gt_tri_orig_for_bev[:, :, 0].reshape(-1)
                    gt_v = gt_tri_orig_for_bev[:, :, 1].reshape(-1)
                    Xg, Yg, Zg, Vg = _bilinear_lut_xyz(
                        lut_data,
                        gt_u,
                        gt_v,
                        min_valid_corners=int(args.lut_min_corners),
                        boundary_eps=float(args.lut_boundary_eps),
                    )
                    Vg = Vg.reshape(-1, 3)
                    good_gt = np.all(Vg, axis=1)
                    Xg = Xg.reshape(-1, 3)
                    Yg = Yg.reshape(-1, 3)
                    gt_tris_bev = np.stack([Xg, Yg], axis=-1).astype(np.float32)
                    gt_tris_bev *= float(args.bev_scale)
                    gt_tris_bev = gt_tris_bev[good_gt]
                else:
                    gt_tris_bev = np.zeros((0, 3, 2), dtype=np.float32)

            else:
                # Homography 방식
                H_img2ground = load_homography(args.calib_dir, name, homography_cache, invert=args.invert_calib)
                if H_img2ground is None:
                    missing_h_names.add(os.path.splitext(name)[0])
                    gt_tris_bev = np.zeros((0, 3, 2), dtype=np.float32)
                else:
                    if pred_tris_orig:
                        pred_stack_orig = np.asarray(pred_tris_orig, dtype=np.float64)
                        pred_tris_bev = apply_homography(pred_stack_orig, H_img2ground)
                        pred_tris_bev = pred_tris_bev * float(args.bev_scale)
                        for det, tri_bev in zip(dets, pred_tris_bev):
                            if not np.all(np.isfinite(tri_bev)):
                                continue
                            poly_bev = poly_from_tri(tri_bev)
                            props = compute_bev_properties_homography(tri_bev)
                            if props is None:
                                continue
                            center, length, width, yaw, front_edge = props

                            if not _sane_dims(length, width, args):
                                continue

                            bev_dets.append({
                                "score": float(det["score"]),
                                "tri": tri_bev,
                                "poly": poly_bev,
                                "center": center,
                                "length": length,
                                "width": width,
                                "yaw": yaw,
                                "front_edge": front_edge,
                                "cz": 0.0,
                                "pitch": 0.0,
                                "roll": 0.0,
                                "class_id": int(det.get("class_id", det.get("cls", 0))),
                            })

                    # GT → BEV (Homography)
                    if gt_tri_orig_for_bev.size > 0:
                        gt_tris_bev = apply_homography(
                            gt_tri_orig_for_bev.astype(np.float64), H_img2ground
                        )
                        gt_tris_bev = (gt_tris_bev * float(args.bev_scale)).astype(np.float32)
                    else:
                        gt_tris_bev = np.zeros((0, 3, 2), dtype=np.float32)

            # BEV 시각화/라벨 저장
            bev_img_path = os.path.join(out_bev_img_dir, name)
            draw_bev_visualization(bev_dets, None, bev_img_path, f"{name} | Pred BEV")

            bev_mix_path = os.path.join(out_bev_mix_dir, name)
            draw_bev_visualization(bev_dets, gt_tris_bev, bev_mix_path, f"{name} | Pred & GT BEV")

            bev_label_path = os.path.join(out_bev_lab_dir, os.path.splitext(name)[0] + ".txt")
            write_bev_labels(bev_label_path, bev_dets, write_3d=bool(args.bev_label_3d))

            total_gt_bev += gt_tris_bev.shape[0]
            if gt_tris_bev.shape[0] > 0 and len(bev_dets) > 0:
                records_bev, _ = evaluate_single_image_bev(
                    bev_dets, gt_tris_bev, iou_thr=args.eval_iou_thr
                )
                metric_records_bev.extend(records_bev)

    # 전체 메트릭 출력
    if do_eval_2d:
        metrics = compute_detection_metrics(metric_records, total_gt)
        print("== 2D Eval (dataset-wide) ==")
        print("Precision:  {:.4f}".format(metrics["precision"]))
        print("Recall:     {:.4f}".format(metrics["recall"]))
        print("mAP@50:     {:.4f}".format(metrics["map50"]))
        print("mAOE(deg):  {:.2f}".format(metrics["mAOE_deg"]))

        if len(total_gt_by_class) > 1:
            per_cls = compute_detection_metrics_per_class(metric_records, total_gt_by_class)
            for cls_id, m in per_cls.items():
                print(f"  [class {cls_id}] Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
                      f"AP@50={m['map50']:.4f}  AOE={m['mAOE_deg']:.2f}°")

    if use_bev:
        metrics_bev = (
            compute_detection_metrics(metric_records_bev, total_gt_bev)
            if (total_gt_bev > 0 or metric_records_bev)
            else None
        )
        if metrics_bev is not None:
            print("== BEV Eval (dataset-wide) ==")
            print("Precision:  {:.4f}".format(metrics_bev["precision"]))
            print("Recall:     {:.4f}".format(metrics_bev["recall"]))
            print("APbev@50:   {:.4f}".format(metrics_bev["map50"]))
            print("mAOE_bev:   {:.2f}".format(metrics_bev["mAOE_deg"]))
        else:
            print("[Info] BEV 평가는 GT 또는 유효 매칭이 부족해 계산하지 않았습니다.")
    else:
        print("[Info] BEV 경로 미지정으로 BEV 라벨/시각화/평가를 생략했습니다.")

    if missing_h_names:
        print(f"[Warn] Homography missing for {len(missing_h_names)} images: {sorted(missing_h_names)[:5]}...")

    print("Done.")


if __name__ == "__main__":
    main()