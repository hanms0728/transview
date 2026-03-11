import math
from typing import List, Dict, Any, Sequence

import cv2
import numpy as np
import torch
import torchvision

from src.geometry_utils import parallelogram_from_triangle, aabb_of_poly4, iou_aabb_xywh, polygon_area


def iou_polygon(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    try:
        pa = poly_a.astype(np.float32)
        pb = poly_b.astype(np.float32)
        inter_area, _ = cv2.intersectConvexConvex(pa, pb)
        if inter_area <= 0:
            return 0.0
        ua = polygon_area(pa)
        ub = polygon_area(pb)
        union = ua + ub - inter_area
        return float(inter_area / max(union, 1e-9))
    except Exception:
        xa1, ya1 = poly_a[:, 0].min(), poly_a[:, 1].min()
        xa2, ya2 = poly_a[:, 0].max(), poly_a[:, 1].max()
        xb1, yb1 = poly_b[:, 0].min(), poly_b[:, 1].min()
        xb2, yb2 = poly_b[:, 0].max(), poly_b[:, 1].max()
        inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
        inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
        inter = inter_w * inter_h
        ua = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        ub = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = ua + ub - inter
        return float(inter / max(union, 1e-9))

def orientation_from_triangle(tri: np.ndarray) -> float:
    tri = np.asarray(tri)
    if tri.shape[0] < 3:
        return 0.0
    vec = _orientation_vec(tri)
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    angle = math.atan2(float(vec[1]), float(vec[0]))
    return angle % math.pi

def orientation_error_deg(pred_tri: np.ndarray, gt_tri: np.ndarray) -> float:
    ap = orientation_from_triangle(pred_tri)
    ag = orientation_from_triangle(gt_tri)
    diff = abs(ap - ag)
    diff = min(diff, math.pi - diff)
    return math.degrees(diff)

def _aabb_metrics(boxA_xywh, boxB_xywh):
    iou = iou_aabb_xywh(boxA_xywh, boxB_xywh)

    ax0, ay0, aw, ah = boxA_xywh
    bx0, by0, bw, bh = boxB_xywh
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih

    areaA = max(0.0, aw) * max(0.0, ah)
    areaB = max(0.0, bw) * max(0.0, bh)
    ios = inter / max(min(areaA, areaB), 1e-9)  # Intersection over Smaller

    return iou, ios


def _nms_iou_or_ios(dets, iou_thr=0.5, contain_thr=None, topk=300):
    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets_sorted:
        boxA = aabb_of_poly4(d["poly4"]) 
        suppress = False
        for k in keep:
            boxB = aabb_of_poly4(k["poly4"]) 
            iou, ios = _aabb_metrics(boxA, boxB)
            if (iou >= iou_thr) or (contain_thr is not None and ios >= contain_thr):
                suppress = True
                break
        if not suppress:
            keep.append(d)
        if len(keep) >= topk:
            break
    return keep


def _resolve_conf_threshold(conf_cfg, cam_id) -> float:
    if isinstance(conf_cfg, dict):
        if cam_id in conf_cfg:
            return float(conf_cfg[cam_id])
        cam_key = str(cam_id)
        if cam_key in conf_cfg:
            return float(conf_cfg[cam_key])
        if conf_cfg:
            return float(next(iter(conf_cfg.values())))
        raise ValueError("conf_th dictionary is empty")
    return float(conf_cfg)


def _orientation_vec(tri: np.ndarray) -> np.ndarray:
    """p0→(p1,p2 중점) 방향벡터. 너무 짧으면 p1→p2로 대체."""
    mid = 0.5 * (tri[1] + tri[2])
    v = mid - tri[0]
    if np.linalg.norm(v) < 1e-6:
        v = tri[2] - tri[1]
    return v


def normalize_triangle(tri: np.ndarray, ref_dir: np.ndarray = None):
    """삼각형 꼭짓점 순서 통일 + ref_dir 기준 방향 교정"""
    tri = np.asarray(tri, dtype=np.float32).copy()
    if tri.shape != (3, 2):
        raise ValueError(f"tri shape must be (3,2), got {tri.shape}")

    area = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if area < 0:
        tri[[1, 2]] = tri[[2, 1]]

    if ref_dir is not None:
        v = _orientation_vec(tri)
        if np.dot(v, ref_dir) < 0:
            tri[[1, 2]] = tri[[2, 1]]
            v = -v
        ref_dir = v
    else:
        ref_dir = _orientation_vec(tri)
    return tri, ref_dir


class EMATracker:
    """IoU 매칭 + EMA 스무딩 트래커"""
    def __init__(self, iou_thresh: float = 0.3, alpha: float = 0.7, max_miss: int = 5):
        self.iou_thresh = float(iou_thresh)
        self.alpha = float(alpha)
        self.max_miss = int(max_miss)
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_id = 1

    def _poly4_iou(self, poly_a: np.ndarray, poly_b: np.ndarray) -> float:
        xa, ya, wa, ha = aabb_of_poly4(poly_a)
        xb, yb, wb, hb = aabb_of_poly4(poly_b)
        return float(iou_aabb_xywh((xa, ya, wa, ha), (xb, yb, wb, hb)))

    def _get_poly(self, det: dict) -> np.ndarray:
        if "poly4" in det:
            return np.asarray(det["poly4"], dtype=np.float32)
        if "tri" in det:
            tri = np.asarray(det["tri"], dtype=np.float32)
            return parallelogram_from_triangle(tri[0], tri[1], tri[2])
        raise ValueError("det must contain 'poly4' or 'tri'")

    def update(self, dets: list) -> list:
        det_polys = [self._get_poly(d) for d in dets]
        det_tris = []
        for d in dets:
            tri = np.asarray(d["tri"], dtype=np.float32) if "tri" in d else None
            det_tris.append(tri)

        track_ids = list(self.tracks.keys())
        iou_mat = np.zeros((len(track_ids), len(dets)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            poly_t = self.tracks[tid]["poly4"]
            for j, poly_d in enumerate(det_polys):
                iou_mat[i, j] = self._poly4_iou(poly_t, poly_d)

        matched_det = set()
        matched_track = set()
        pairs = []
        while True:
            if iou_mat.size == 0:
                break
            i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[i, j] < self.iou_thresh:
                break
            if track_ids[i] in matched_track or j in matched_det:
                iou_mat[i, j] = -1
                continue
            pairs.append((track_ids[i], j))
            matched_track.add(track_ids[i])
            matched_det.add(j)
            iou_mat[i, j] = -1

        for tid, j in pairs:
            tri_new = det_tris[j]
            poly_new = det_polys[j]
            track = self.tracks[tid]
            if tri_new is not None:
                tri_new, ref_dir = normalize_triangle(tri_new, track.get("ref_dir"))
                track["ref_dir"] = ref_dir
                tri_smoothed = self.alpha * track["tri"] + (1 - self.alpha) * tri_new
                track["tri"] = tri_smoothed
                track["poly4"] = parallelogram_from_triangle(tri_smoothed[0], tri_smoothed[1], tri_smoothed[2])
            else:
                track["poly4"] = self.alpha * track["poly4"] + (1 - self.alpha) * poly_new
            track["miss"] = 0
            track["last_score"] = float(dets[j].get("score", track.get("last_score", 0.0)))
            track["last_cls"] = dets[j].get("class_id", dets[j].get("cls", track.get("last_cls", 0)))

        for j, tri_new in enumerate(det_tris):
            if j in matched_det:
                continue
            poly_new = det_polys[j]
            if tri_new is not None:
                tri_new, ref_dir = normalize_triangle(tri_new)
                poly_new = parallelogram_from_triangle(tri_new[0], tri_new[1], tri_new[2])
            else:
                ref_dir = None
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "tri": tri_new if tri_new is not None else None,
                "poly4": poly_new,
                "ref_dir": ref_dir,
                "miss": 0,
                "last_score": float(dets[j].get("score", 0.0)),
                "last_cls": dets[j].get("class_id", dets[j].get("cls", 0)),
            }

        to_delete = []
        for tid in self.tracks:
            if tid in matched_track:
                continue
            self.tracks[tid]["miss"] += 1
            if self.tracks[tid]["miss"] > self.max_miss:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)

        out = []
        for tid, tr in self.tracks.items():
            d_out = {
                "track_id": tid,
                "poly4": tr["poly4"],
                "score": tr.get("last_score", 0.0),
                "class_id": tr.get("last_cls", 0),
            }
            if tr.get("tri") is not None:
                d_out["tri"] = tr["tri"]
            out.append(d_out)
        return out


def _decode_predictions_impl(
    cam_id,
    outputs,
    strides,
    clip_cells=None,
    conf_th=0.15,
    nms_iou=0.5,
    topk=300,
    contain_thr=0.7,     #작은 객체의 큰 객체 대비 겹침 정도
    score_mode="obj",      # "obj" | "cls" | "obj*cls"
    use_gpu_nms=False      # True면 torchvision.ops.nms 사용
):
    """
    outputs: [(reg,obj,cls)] * L, 각 텐서 shape = (B, C, Hs, Ws)
    - reg는 셀 단위 오프셋을 예측 (train.py와 동일)
    - centers = ((x+0.5)*s, (y+0.5)*s)
    - 최종 점: centers + reg*stride  (px)
    """
    assert score_mode in ("obj", "cls", "obj*cls")
    B = outputs[0][0].shape[0]
    batch_results = []

    for b in range(B):
        dets = []
        boxes_for_nms = []
        scores_for_nms = []

        for l, (reg, obj, cls) in enumerate(outputs):
            stride = float(strides[l]) if not isinstance(strides[l], torch.Tensor) else float(strides[l].item())

            # 점수 맵 만들기
            obj_map = torch.sigmoid(obj[b, 0])  # (Hs,Ws)
            cls_logits = cls[b]
            if cls_logits.shape[0] == 1:
                cls_score_map = torch.sigmoid(cls_logits[0])
                cls_id_map = torch.zeros_like(cls_score_map, dtype=torch.long)
            else:
                cls_prob = torch.softmax(cls_logits, dim=0)
                cls_score_map, cls_id_map = torch.max(cls_prob, dim=0)

            if score_mode == "obj":
                score_map = obj_map
            elif score_mode == "cls":
                score_map = cls_score_map
            else:  # "obj*cls"
                score_map = obj_map * cls_score_map

            thr = _resolve_conf_threshold(conf_th, cam_id)
            keep = score_map > thr
            if keep.sum().item() == 0:
                continue

            # 좌표 준비
            keep_idx = keep.nonzero(as_tuple=False)  # (K,2) [y,x]
            ys = keep_idx[:, 0].float()
            xs = keep_idx[:, 1].float()
            scores = score_map[keep]
            cls_scores = cls_score_map[keep]
            cls_ids = cls_id_map[keep]

            # 회귀 맵: (Hs,Ws,3,2)
            reg_map = reg[b].detach().permute(1, 2, 0).reshape(obj_map.shape[0], obj_map.shape[1], 3, 2)
            pred_off = reg_map[keep]  # (K,3,2) in cells

            # (옵션) tanh 클립
            if clip_cells is not None:
                R = float(clip_cells)
                pred_off = R * torch.tanh(pred_off / max(1e-6, R))

            # anchors (px) & 절대 좌표 (px)
            centers = torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1)  # (K,2)
            pred_tri = centers[:, None, :] + pred_off * stride  # (K,3,2) in px

            # dets 작성 + NMS용 AABB 수집
            tri_np = pred_tri.cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            cls_scores_np = cls_scores.detach().cpu().numpy()
            cls_ids_np = cls_ids.detach().cpu().numpy()
            for tri_pts, sc, cls_sc, cls_idx in zip(tri_np, scores_np, cls_scores_np, cls_ids_np):
                poly4 = parallelogram_from_triangle(tri_pts[0], tri_pts[1], tri_pts[2])
                dets.append({
                    "score": float(sc),
                    "cls_score": float(cls_sc),
                    "class_id": int(cls_idx),
                    "cls": int(cls_idx),
                    "poly4": poly4,
                    "tri": tri_pts
                })
                # NMS용 xyxy
                x0, y0 = poly4[:,0].min(), poly4[:,1].min()
                x1, y1 = poly4[:,0].max(), poly4[:,1].max()
                boxes_for_nms.append([x0, y0, x1, y1])
                scores_for_nms.append(float(sc))

        # NMS
        if not dets:
            batch_results.append([])
            continue

        if use_gpu_nms and len(dets) > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            boxes_t = torch.tensor(boxes_for_nms, dtype=torch.float32, device=device)
            scores_t = torch.tensor(scores_for_nms, dtype=torch.float32, device=device)
            keep_idx = torchvision.ops.nms(boxes_t, scores_t, nms_iou)
            keep_idx = keep_idx[:topk].detach().cpu().tolist()
            dets = [dets[i] for i in keep_idx]
            # ← GPU NMS 이후 "포함율" 후처리(선택)
            if contain_thr is not None:
                dets = _nms_iou_or_ios(dets, iou_thr=1.1, contain_thr=contain_thr, topk=topk)
                # iou_thr=1.1로 두면 사실상 IoU 조건은 무시되고 IoS만 적용하는 후처리
                    
            batch_results.append(dets)
        else:
            dets_nms = _nms_iou_or_ios(
                dets,
                iou_thr=nms_iou,
                contain_thr=(float(contain_thr) if isinstance(contain_thr, (int, float)) else None),
                topk=topk
            )
            batch_results.append(dets_nms)

    return batch_results


def decode_predictions(*args, **kwargs):
    """cam_id 유무에 따라 신/구 시그니처 모두 호환"""

    positional_opt_names: Sequence[str] = (
        "clip_cells",
        "conf_th",
        "nms_iou",
        "topk",
        "contain_thr",
        "score_mode",
        "use_gpu_nms",
    )

    if len(args) >= 3 and not isinstance(args[0], (list, tuple)):
        # 신규 시그니처(cam_id, outputs, strides, ...)
        cam_id = args[0]
        outputs = args[1]
        strides = args[2]
        remaining = list(args[3:])
    else:
        # 구 버전(outputs, strides, ...)
        cam_id = kwargs.pop("cam_id", 0)
        outputs = args[0]
        strides = args[1]
        remaining = list(args[2:])

    for name, value in zip(positional_opt_names, remaining):
        kwargs.setdefault(name, value)

    return _decode_predictions_impl(cam_id, outputs, strides, **kwargs)

def evaluate_single_image(preds, gt_tris, gt_classes=None, iou_thr=0.5):
    gt_tris = np.asarray(gt_tris)
    num_gt = gt_tris.shape[0]
    if gt_classes is not None:
        gt_classes = np.asarray(gt_classes).reshape(-1)
        if gt_classes.shape[0] != num_gt:
            raise ValueError("gt_classes length must match gt_tris length")
    preds_sorted = sorted(preds, key=lambda d: d["score"], reverse=True)
    if num_gt == 0:
        records = [(det["score"], 0, 0.0, None, int(det.get("class_id", det.get("cls", 0)))) for det in preds_sorted]
        return records, 0

    gt_polys = [parallelogram_from_triangle(tri[0], tri[1], tri[2]) for tri in gt_tris]
    matched = [False] * num_gt
    records = []

    for det in preds_sorted:
        pred_poly = np.asarray(det["poly4"], dtype=np.float32)
        det_cls = int(det.get("class_id", det.get("cls", 0)))
        best_iou = 0.0
        best_idx = -1
        for idx, gt_poly in enumerate(gt_polys):
            if matched[idx]:
                continue
            if gt_classes is not None and det_cls != int(gt_classes[idx]):
                continue
            iou = iou_polygon(pred_poly, gt_poly.astype(np.float32))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            orient_err = orientation_error_deg(det["tri"], gt_tris[best_idx])
            records.append((det["score"], 1, best_iou, orient_err, det_cls))
        else:
            records.append((det["score"], 0, best_iou, None, det_cls))

    return records, sum(matched)

def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def compute_detection_metrics(records, total_gt):
    if not records:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "mAOE_deg": float('nan')}

    records.sort(key=lambda r: r[0], reverse=True)
    tps = np.array([r[1] for r in records], dtype=np.float32)
    fps = 1.0 - tps
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    denom = np.maximum(tp_cum + fp_cum, 1e-9)
    precisions = tp_cum / denom

    if total_gt > 0:
        recalls = tp_cum / total_gt
    else:
        recalls = np.zeros_like(tp_cum)

    precision = float(precisions[-1]) if precisions.size > 0 else 0.0
    recall = float(recalls[-1]) if total_gt > 0 and recalls.size > 0 else 0.0
    map50 = _average_precision(recalls, precisions) if total_gt > 0 else 0.0

    orient_errors = [r[3] for r in records if r[1] == 1 and r[3] is not None]
    mAOE = float(np.mean(orient_errors)) if orient_errors else float('nan')

    return {"precision": precision, "recall": recall, "map50": map50, "mAOE_deg": mAOE}


def compute_detection_metrics_per_class(records, total_gt_by_class):
    """클래스별 detection metrics 계산.

    Args:
        records: list of (score, tp, iou, orient_err, class_id)
        total_gt_by_class: dict {class_id: count}

    Returns:
        dict {class_id: {precision, recall, map50, mAOE_deg}}
    """
    from collections import defaultdict
    records_by_class = defaultdict(list)
    for r in records:
        cls_id = r[4] if len(r) > 4 else 0
        records_by_class[cls_id].append(r)

    all_classes = set(records_by_class.keys()) | set(total_gt_by_class.keys())
    results = {}
    for cls_id in sorted(all_classes):
        cls_records = records_by_class.get(cls_id, [])
        cls_gt = total_gt_by_class.get(cls_id, 0)
        results[cls_id] = compute_detection_metrics(cls_records, cls_gt)
    return results
