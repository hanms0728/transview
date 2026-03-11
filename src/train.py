# -*- coding: utf-8 -*-
# train.py (mAP@50 베스트 저장 + CSV 로깅 + 저장경로/파일명 커스텀 + 양쪽 스케줄러)
# + 재귀 탐색(A/B 레이아웃 자동 지원) + 시퀀스 윈도우 + ConvLSTM/ConvGRU (헤드 직전 시계열 모듈)
# + ONNX 내보내기(에포크마다): temporal none/GRU/LSTM(+last scale) 지원
import os
import csv
import cv2
import math
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from tqdm import tqdm
import torchvision
try:
    import albumentations as A
except ImportError:
    A = None

from src.geometry_utils import parallelogram_from_triangle as _para
from src.geometry_utils import aabb_of_poly4, iou_aabb_xywh, tiny_filter_on_dets, polygon_area
from src.evaluation_utils import decode_predictions, evaluate_single_image, compute_detection_metrics

# matmul/TF32 설정
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

torch.backends.cudnn.benchmark = True

# 시드 고정
def set_seed(s=42):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed(42)

# AMP 호환
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def amp_autocast():
        return _autocast("cuda", enabled=torch.cuda.is_available())
    GradScaler = _GradScaler
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as GradScaler
    def amp_autocast():
        return _autocast(enabled=torch.cuda.is_available())

# Geometry helpers (라벨 어사인용)
def _point_in_triangle(px, py, tri):
    Ax, Ay = tri[0, 0], tri[0, 1]
    Bx, By = tri[1, 0], tri[1, 1]
    Cx, Cy = tri[2, 0], tri[2, 1]
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    d1 = sign(px, py, Ax, Ay, Bx, By)
    d2 = sign(px, py, Bx, By, Cx, Cy)
    d3 = sign(px, py, Cx, Cy, Ax, Ay)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    inside = ~(has_neg & has_pos)
    return inside

def _point_to_segment_dist(px, py, x1, y1, x2, y2, eps=1e-9):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx*vx + vy*vy + eps
    t = (wx*vx + wy*vy) / vv
    t = torch.clamp(t, 0.0, 1.0)
    projx = x1 + t * vx; projy = y1 + t * vy
    dx = px - projx; dy = py - projy
    return torch.sqrt(dx*dx + dy*dy + 1e-12)

def _point_to_triangle_dist(px, py, tri):
    Ax, Ay = tri[0, 0], tri[0, 1]
    Bx, By = tri[1, 0], tri[1, 1]
    Cx, Cy = tri[2, 0], tri[2, 1]
    d1 = _point_to_segment_dist(px, py, Ax, Ay, Bx, By)
    d2 = _point_to_segment_dist(px, py, Bx, By, Cx, Cy)
    d3 = _point_to_segment_dist(px, py, Cx, Cy, Ax, Ay)
    return torch.minimum(d1, torch.minimum(d2, d3))

# Dataset (레이아웃 A/B 자동 지원)
class ParallelogramDataset(Dataset):
    """A: <root>/images & labels, B: <root>/<vid>/images & labels 구조 지원."""
    def __init__(self, root, target_size=(1088, 1920), transform=None, data_layout="auto"):
        super().__init__()
        self.Ht, self.Wt = target_size
        self.transform = transform
        self.path_map = {}   # rel -> (img_abs, lab_abs)
        self.img_files = []  # rel list (grouping/정렬/시퀀스에 사용)

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root not found: {root}")

        def is_layout_A():
            return os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "labels"))

        def is_layout_B():
            for d in os.listdir(root):
                vid_dir = os.path.join(root, d)
                if not os.path.isdir(vid_dir): 
                    continue
                if os.path.isdir(os.path.join(vid_dir, "images")) and os.path.isdir(os.path.join(vid_dir, "labels")):
                    return True
            return False

        # 레이아웃 결정
        if data_layout == "A" or (data_layout == "auto" and is_layout_A()):
            mode = "A"
        elif data_layout == "B" or (data_layout == "auto" and is_layout_B()):
            mode = "B"
        else:
            raise RuntimeError(
                "데이터 레이아웃을 판별할 수 없습니다.\n"
                "A: <root>/images & <root>/labels  또는 "
                "B: <root>/<vid>/{images,labels} 구조인지 확인하세요."
            )

        self.mode = mode
        self.root = root

        # 파일 수집
        if mode == "A":
            img_root = os.path.join(root, "images")
            lab_root = os.path.join(root, "labels")
            if not (os.path.isdir(img_root) and os.path.isdir(lab_root)):
                raise FileNotFoundError("레이아웃 A: images/ 또는 labels/ 폴더가 없습니다.")
            for r, _, files in os.walk(img_root):
                for f in files:
                    if not f.lower().endswith(('.jpg','.jpeg','.png')): 
                        continue
                    img_abs = os.path.join(r, f)
                    rel = os.path.relpath(img_abs, img_root)                # e.g., videoA/0001.jpg
                    lab_abs = os.path.join(lab_root, os.path.splitext(rel)[0] + ".txt")
                    if os.path.exists(lab_abs):
                        rel_norm = rel.replace("\\", "/")
                        self.path_map[rel_norm] = (img_abs, lab_abs)
                        self.img_files.append(rel_norm)
        else:  # mode == "B"
            for d in os.listdir(root):
                vid_dir = os.path.join(root, d)
                if not os.path.isdir(vid_dir): 
                    continue
                img_root = os.path.join(vid_dir, "images")
                lab_root = os.path.join(vid_dir, "labels")
                if not (os.path.isdir(img_root) and os.path.isdir(lab_root)):
                    continue
                for r, _, files in os.walk(img_root):
                    for f in files:
                        if not f.lower().endswith(('.jpg','.jpeg','.png')): 
                            continue
                        img_abs = os.path.join(r, f)
                        rel_local = os.path.relpath(img_abs, img_root)      # e.g., 0001.jpg or sub/0001.jpg
                        rel = os.path.join(d, rel_local).replace("\\", "/")  # e.g., scene_001/0001.jpg
                        lab_abs = os.path.join(lab_root, os.path.splitext(rel_local)[0] + ".txt")
                        if os.path.exists(lab_abs):
                            self.path_map[rel] = (img_abs, lab_abs)
                            self.img_files.append(rel)

        # 정렬 (숫자 우선 → 문자열)
        def key_fn(nm):
            s = Path(nm).stem
            digits = "".join([c for c in s if c.isdigit()])
            return (int(digits) if digits.isdigit() else 0, nm)
        self.img_files.sort(key=key_fn)

        print(f"[Dataset-{self.mode}] matched pairs = {len(self.img_files)}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        rel = self.img_files[idx]
        img_path, lab_path = self.path_map[rel]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        H0, W0 = img_bgr.shape[:2]

        img_bgr = cv2.resize(img_bgr, (self.Wt, self.Ht), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        points_list, labels_list = [], []
        with open(lab_path,'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 7: continue
                cls_id, p0x,p0y,p1x,p1y,p2x,p2y = parts
                sX = self.Wt / W0; sY = self.Ht / H0
                p0 = [float(p0x)*sX, float(p0y)*sY]
                p1 = [float(p1x)*sX, float(p1y)*sY]
                # 에디터 붙여넣기 오류 방지해도 최종은 아래 p2만 유효
                p0 = [float(p0x)*sX, float(p0y)*sY]
                p1 = [float(p1x)*sX, float(p1y)*sY]
                p2 = [float(p2x)*sX, float(p2y)*sY]
                # NaN 라벨 방어
                if any([np.isnan(v) or np.isinf(v) for v in (*p0, *p1, *p2)]):
                    continue
                points_list.append([p0,p1,p2])
                try:
                    cls_idx = int(float(cls_id))
                except (ValueError, TypeError):
                    cls_idx = 0
                labels_list.append(cls_idx)

        img_rgb, points_list = apply_transform_if_available(img_rgb, points_list, self.transform, self.Wt, self.Ht)
        img = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
        targets = {
            "points": torch.tensor(points_list, dtype=torch.float32),
            "labels": torch.tensor(labels_list, dtype=torch.long)
        }
        name = rel
        return img, targets, name

def collate_fn(batch):
    imgs  = torch.stack([b[0] for b in batch],0)
    tgts  = [b[1] for b in batch]
    names = [b[2] for b in batch]
    return imgs, tgts, names

# 시퀀스 윈도우 래퍼
class SeqWindowDataset(Dataset):
    """T 프레임 시퀀스 윈도우 반환. grouping: auto|by_subdir|by_prefix|flat"""
    def __init__(self, base: ParallelogramDataset, grouping="auto", seq_len=4, seq_stride=1):
        super().__init__()
        self.base = base
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.grouping = grouping

        names = list(base.img_files)

        def has_subdirs():
            return any("/" in nm for nm in names)

        mode = grouping
        if grouping == "auto":
            mode = "by_subdir" if has_subdirs() else "by_prefix"

        groups = {}
        if mode == "by_subdir":
            for nm in names:
                parts = Path(nm).parts
                vid = parts[0] if len(parts) >= 2 else "_root"
                groups.setdefault(vid, []).append(nm)
        elif mode == "by_prefix":
            for nm in names:
                stem = Path(nm).stem
                if "_" in stem:
                    vid = stem.split("_")[0]
                elif "-" in stem:
                    vid = stem.split("-")[0]
                else:
                    vid = "ALL"
                groups.setdefault(vid, []).append(nm)
        else:  # "flat"
            groups["ALL"] = names

        def key_fn(nm):
            s = Path(nm).stem
            digits = "".join([c for c in s if c.isdigit()])
            return (int(digits) if digits.isdigit() else 0, s)
        for k in list(groups.keys()):
            groups[k] = sorted(groups[k], key=key_fn)

        self.windows = []  # [(vid, [rel names...])]
        T = self.seq_len
        S = self.seq_stride
        for vid, lst in groups.items():
            if len(lst) < T:
                continue
            for st in range(0, len(lst) - T + 1, S):
                win = lst[st:st+T]
                if len(win) == T:
                    self.windows.append((vid, win))

        if len(self.windows) == 0:
            for vid, lst in groups.items():
                for nm in lst:
                    self.windows.append((vid, [nm]))

        print(f"[SeqWindowDataset] groups={len(groups)} windows={len(self.windows)} (T={self.seq_len}, S={self.seq_stride}, mode={mode})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        vid, name_list = self.windows[idx]
        imgs = []; tgts = []; names = []
        for rel in name_list:
            img_abs, lab_abs = self.base.path_map[rel]

            img_bgr = cv2.imread(img_abs)
            if img_bgr is None:
                raise FileNotFoundError(img_abs)
            H0, W0 = img_bgr.shape[:2]
            img_bgr = cv2.resize(img_bgr, (self.base.Wt, self.base.Ht), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            points_list, labels_list = [], []
            with open(lab_abs,'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 7: continue
                    cls_id, p0x,p0y,p1x,p1y,p2x,p2y = parts
                    sX = self.base.Wt / W0; sY = self.base.Ht / H0
                    p0 = [float(p0x)*sX, float(p0y)*sY]
                    p1 = [float(p1x)*sX, float(p1y)*sY]
                    p2 = [float(p2x)*sX, float(p2y)*sY]
                    if any([np.isnan(v) or np.isinf(v) for v in (*p0, *p1, *p2)]):
                        continue
                    points_list.append([p0,p1,p2])
                    try:
                        cls_idx = int(float(cls_id))
                    except (ValueError, TypeError):
                        cls_idx = 0
                    labels_list.append(cls_idx)
            img_rgb, points_list = apply_transform_if_available(
                img_rgb, points_list, self.base.transform, self.base.Wt, self.base.Ht
            )
            img = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
            targets = {
                "points": torch.tensor(points_list, dtype=torch.float32),
                "labels": torch.tensor(labels_list, dtype=torch.long)
            }
            imgs.append(img); tgts.append(targets); names.append(rel)

        imgs = torch.stack(imgs, dim=0)  # (T,3,H,W)
        return imgs, tgts, names, vid

def collate_seq_fn(batch):
    T = batch[0][0].shape[0]
    imgs = torch.stack([b[0] for b in batch], dim=0)  # (B,T,3,H,W)
    tgts = [b[1] for b in batch]
    names= [b[2] for b in batch]
    vids = [b[3] for b in batch]
    return imgs, tgts, names, vids, T


def apply_transform_if_available(img_rgb: np.ndarray, points_list: List[List[List[float]]],
                                 transform, Wt: int, Ht: int):
    if transform is None:
        return img_rgb, points_list
    flat_points = [tuple(pt) for tri in points_list for pt in tri]
    aug = transform(image=img_rgb, keypoints=flat_points)
    aug_pts = aug["keypoints"]
    if len(flat_points) == 0 or (aug_pts and len(aug_pts) == len(flat_points)):
        valid = True
        new_points = []
        if len(flat_points) > 0:
            for i in range(0, len(aug_pts), 3):
                tri = []
                inside = 0
                for k in range(3):
                    x, y = aug_pts[i + k]
                    tri.append([float(x), float(y)])
                    if 0.0 <= x < Wt and 0.0 <= y < Ht:
                        inside += 1
                if inside >= 2:
                    new_points.append(tri)
                else:
                    valid = False
                    break
        if valid:
            img_rgb = aug["image"]
            if len(flat_points) > 0:
                points_list = new_points
    return img_rgb, points_list


def build_default_train_augment(target_size):
    if A is None:
        raise ImportError("Albumentations is required for train augment. Install via `pip install albumentations`.")
    Ht, Wt = target_size
    rrc_kwargs = dict(scale=(0.85, 1.0), ratio=(0.8, 1.2), p=0.4)
    try:
        rrc = A.RandomResizedCrop(height=Ht, width=Wt, **rrc_kwargs)
    except (TypeError, ValueError):
        rrc = A.RandomResizedCrop(size=(Ht, Wt), **rrc_kwargs)
    return A.Compose(
        [
            rrc,
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.4),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.02, 0.02),
                rotate=(-5, 5),
                shear=(-2, 2),
                p=0.6
            ),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, pad_mode=cv2.BORDER_REFLECT_101, p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )

# DSI (Vision Teacher Alignment)
class VisionTeacherWrapper(nn.Module):
    SUPPORTED = {"vit_b_16", "vit_l_16"}

    def __init__(self, arch="vit_b_16", pretrained=True, checkpoint=None,
                 resize_hw=None, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), device="cpu"):
        super().__init__()
        arch = arch.lower()
        if arch not in self.SUPPORTED:
            raise ValueError(f"Unsupported teacher arch '{arch}'. Supported: {sorted(self.SUPPORTED)}")
        self.resize_hw = resize_hw
        self.teacher = self._build_teacher(arch, pretrained, checkpoint).to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        patch_embed = getattr(self.teacher, "conv_proj", None)
        if patch_embed is None:
            raise RuntimeError("Unable to locate patch embedding in teacher model.")
        self.patch_size = patch_embed.kernel_size[0]
        self.embed_dim = getattr(self.teacher, "hidden_dim", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(self.teacher, "embed_dim", None)
        if self.embed_dim is None:
            raise RuntimeError("Unable to infer teacher embedding dimension.")

        self.register_buffer("img_mean", torch.tensor(mean, dtype=torch.float32).view(1,3,1,1))
        self.register_buffer("img_std", torch.tensor(std, dtype=torch.float32).view(1,3,1,1))

    def _build_teacher(self, arch, pretrained, checkpoint):
        if arch == "vit_b_16":
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
            model = torchvision.models.vit_b_16(weights=weights)
        elif arch == "vit_l_16":
            weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
            model = torchvision.models.vit_l_16(weights=weights)
        else:
            raise ValueError(f"Unsupported teacher arch '{arch}'")
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"[DSI] Loaded teacher checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
        return model

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        target_hw = self.resize_hw if self.resize_hw is not None else (images.shape[2], images.shape[3])
        if self.resize_hw is not None:
            x = F.interpolate(x, size=self.resize_hw, mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        n = x.shape[0]
        x = self.teacher._process_input(x)
        batch_class_token = self.teacher.class_token.expand(n, -1, -1)
        x = torch.cat((batch_class_token, x), dim=1)
        x = self.teacher.encoder(x)
        if hasattr(self.teacher, "ln"):
            x = self.teacher.ln(x)
        elif hasattr(self.teacher.encoder, "ln"):
            x = self.teacher.encoder.ln(x)
        elif hasattr(self.teacher, "norm"):
            x = self.teacher.norm(x)
        x = x[:, 1:, :]
        grid_h = max(1, target_hw[0] // self.patch_size)
        grid_w = max(1, target_hw[1] // self.patch_size)
        if grid_h * grid_w != x.shape[1]:
            side = int(x.shape[1] ** 0.5)
            grid_h = grid_w = max(1, side)
        x = x.permute(0, 2, 1).reshape(n, self.embed_dim, grid_h, grid_w)
        return x


class DeepSemanticInjector(nn.Module):
    def __init__(self, student_channels, arch, pretrained, checkpoint,
                 resize_hw, mean, std, device):
        super().__init__()
        self.teacher = VisionTeacherWrapper(
            arch=arch,
            pretrained=pretrained,
            checkpoint=checkpoint,
            resize_hw=resize_hw,
            mean=mean,
            std=std,
            device=device
        )
        self.projector = nn.Sequential(
            nn.Conv2d(student_channels, self.teacher.embed_dim, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=self.teacher.embed_dim)
        )
        self.device = device

    def forward(self, student_feat, images):
        teacher_feat = self.teacher(images.to(self.device, non_blocking=True))
        teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False)
        proj_student = self.projector(student_feat)
        stud_flat = F.normalize(proj_student.flatten(2).transpose(1, 2), dim=-1)
        teach_flat = F.normalize(teacher_feat.flatten(2).transpose(1, 2), dim=-1)
        cos = torch.sum(stud_flat * teach_flat, dim=-1)
        loss = 1.0 - cos.mean()
        return loss


def module_grad_l1(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().abs().sum().item())
    return total

# ConvRNN: ConvGRU / ConvLSTM (헤드 직전용)
class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, k, padding=p)
        self.conv_h  = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=p)

    def forward(self, x, h):
        if h is None or h.shape[0] != x.size(0) or h.shape[2] != x.size(2) or h.shape[3] != x.size(3):
            h = x.new_zeros((x.size(0), self.conv_h.out_channels, x.size(2), x.size(3)))
        z_r = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = torch.split(z_r, z_r.size(1)//2, dim=1)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * h_tilde
        return h

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)

    def forward(self, x, state):
        h, c = state if state is not None else (None, None)
        if (h is None or c is None or
            h.shape[0] != x.size(0) or h.shape[2] != x.size(2) or h.shape[3] != x.size(3)):
            h = x.new_zeros((x.size(0), self.hid_ch, x.size(2), x.size(3)))
            c = x.new_zeros((x.size(0), self.hid_ch, x.size(2), x.size(3)))
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class TemporalBlock(nn.Module):
    """헤드 직전 시계열 어댑터. 1x1 bottleneck -> ConvRNN -> 1x1 복원."""
    def __init__(self, in_ch, hid_ch=256, layers=1, mode="gru"):
        super().__init__()
        assert mode in ("gru", "lstm")
        self.mode = mode
        self.hid_ch = hid_ch
        self.reduce = nn.Conv2d(in_ch, hid_ch, 1, 1, 0)
        self.restore = nn.Conv2d(hid_ch, in_ch, 1, 1, 0)
        cells = []
        for _ in range(layers):
            if mode == "gru":
                cells.append(ConvGRUCell(hid_ch, hid_ch))
            else:
                cells.append(ConvLSTMCell(hid_ch, hid_ch))
        self.cells = nn.ModuleList(cells)
        self._state = [None for _ in range(layers)]  # GRU: h, LSTM: (h,c)

    def reset_state(self):
        self._state = [None for _ in self._state]

    @torch.no_grad()
    def detach_state(self):
        for i, st in enumerate(self._state):
            if st is None: continue
            if self.mode == "gru":
                self._state[i] = st.detach()
            else:
                h, c = st
                self._state[i] = (h.detach(), c.detach())

    def forward(self, x):
        z = self.reduce(x)
        if self.mode == "gru":
            h = z
            for i, cell in enumerate(self.cells):
                h = cell(h, self._state[i])
                self._state[i] = h
            out = h
        else:
            h = z; state = None
            for i, cell in enumerate(self.cells):
                state = cell(h, self._state[i])
                h = state[0]
                self._state[i] = state
            out = h
        out = self.restore(out)
        return out

    # ONNX용: 외부 상태 입출력 1 step
    def step_with_state(self, x, state_in: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]):
        z = self.reduce(x)
        if self.mode == "gru":
            h = z
            h = self.cells[0](h, state_in)  # 1층만 외부 상태 연결
            out = self.restore(h)
            return out, h
        else:
            h = z
            st = self.cells[0](h, state_in)  # (h,c)
            out = self.restore(st[0])
            return out, st

# Model
class TriHead(nn.Module):
    def __init__(self, in_ch, nc, prior_p=0.20):
        super().__init__()
        self.reg = nn.Conv2d(in_ch, 6, 1, 1, 0)
        self.obj = nn.Conv2d(in_ch, 1, 1, 1, 0)
        self.cls = nn.Conv2d(in_ch, nc, 1, 1, 0)
        prior_b = math.log(prior_p/(1-prior_p))
        nn.init.constant_(self.obj.bias, prior_b)
        nn.init.constant_(self.cls.bias, prior_b)
        nn.init.zeros_(self.reg.weight)
        nn.init.zeros_(self.reg.bias)
    def forward(self, x):
        return self.reg(x), self.obj(x), self.cls(x)

class YOLO11_2_5D(nn.Module):
    def __init__(self, yolo11_path='yolo11m.pt', num_classes=1, img_size=(1088,1920),
                 temporal_mode="none", temporal_hidden=256, temporal_layers=1, temporal_on_scales="last",
                 include_p2=False):
        super().__init__()
        base = YOLO(yolo11_path).model
        self.include_p2 = bool(include_p2)
        self.backbone_neck = base.model[:-1]
        self.save_idx = list(base.save)
        if self.include_p2:
            # layer 2: stride 4 feature (C3k2)
            self.p2_source_idx = 2
            if self.p2_source_idx not in self.save_idx:
                self.save_idx.append(self.p2_source_idx)
            self.save_idx = sorted(set(self.save_idx))
        detect = base.model[-1]
        self.strides = detect.stride
        self.f_indices = detect.f

        # in_chs 산출
        self.backbone_neck.eval()
        with torch.no_grad():
            dummy = torch.zeros(1,3,img_size[0],img_size[1])
            feats_memory = []
            x = dummy
            for m in self.backbone_neck:
                if m.f != -1:
                    x = feats_memory[m.f] if isinstance(m.f,int) else [x if j==-1 else feats_memory[j] for j in m.f]
                x = m(x)
                feats_memory.append(x if m.i in self.save_idx else None)
            feat_list = [feats_memory[i] for i in self.f_indices]
            in_chs = [f.shape[1] for f in feat_list]
            if self.include_p2:
                p2_src = feats_memory[self.p2_source_idx]
                self.p2_src_ch = p2_src.shape[1]
                self.p3_ch = in_chs[0]
        self.backbone_neck.train()

        # heads
        self.p2_reduce = None
        self.p2_lateral = None
        self.p2_upsample = None
        self.p2_fuse = None
        if self.include_p2:
            p2_lat_ch = max(32, self.p3_ch // 2)
            p3_red = max(32, self.p3_ch // 2)
            self.p2_reduce = nn.Sequential(
                nn.Conv2d(self.p3_ch, p3_red, 1, 1, 0, bias=False),
                nn.BatchNorm2d(p3_red),
                nn.SiLU()
            )
            self.p2_lateral = nn.Sequential(
                nn.Conv2d(self.p2_src_ch, p2_lat_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(p2_lat_ch),
                nn.SiLU()
            )
            self.p2_upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.p2_fuse = nn.Sequential(
                nn.Conv2d(p3_red + p2_lat_ch, self.p3_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.p3_ch),
                nn.SiLU()
            )
            in_chs = [self.p3_ch] + in_chs
        self.heads = nn.ModuleList([TriHead(c, num_classes) for c in in_chs])
        self.num_classes = num_classes
        self.deep_feature_channels = in_chs[-1]

        # temporal
        self.temporal_mode = temporal_mode
        self.temporal_on_scales = temporal_on_scales
        if temporal_mode != "none":
            if temporal_on_scales == "all":
                self.temporal = nn.ModuleList([
                    TemporalBlock(c, hid_ch=temporal_hidden, layers=temporal_layers, mode=temporal_mode)
                    for c in in_chs
                ])
            else:
                self.temporal = nn.ModuleList([None for _ in in_chs])
                self.temporal[-1] = TemporalBlock(in_chs[-1], hid_ch=temporal_hidden,
                                                  layers=temporal_layers, mode=temporal_mode)
        else:
            self.temporal = None

        if self.include_p2:
            self.strides = [4.0] + list(self.strides)

    def reset_temporal(self):
        if self.temporal is None: return
        for t in self.temporal:
            if t is not None: t.reset_state()

    @torch.no_grad()
    def detach_temporal(self):
        if self.temporal is None: return
        for t in self.temporal:
            if t is not None: t.detach_state()

    # 내부상태 버전 (학습/일반추론)
    def forward(self, x, use_temporal=True, return_deep_feature=False):
        feats_memory = []
        for m in self.backbone_neck:
            if m.f != -1:
                x = feats_memory[m.f] if isinstance(m.f,int) else [x if j==-1 else feats_memory[j] for j in m.f]
            x = m(x)
            feats_memory.append(x if m.i in self.save_idx else None)
        feat_list = [feats_memory[i] for i in self.f_indices]

        if self.include_p2:
            p3 = feat_list[0]
            p2_src = feats_memory[self.p2_source_idx]
            p3_red = self.p2_reduce(p3)
            p2_lat = self.p2_lateral(p2_src)
            p2_up = self.p2_upsample(p3_red)
            p2 = torch.cat([p2_up, p2_lat], dim=1)
            p2 = self.p2_fuse(p2)
            feat_list = [p2] + feat_list

        if use_temporal and self.temporal is not None:
            feat_new = []
            for idx, f in enumerate(feat_list):
                tb = None
                if self.temporal_on_scales == "all":
                    tb = self.temporal[idx]
                else:
                    if idx == len(feat_list) - 1:
                        tb = self.temporal[-1]
                feat_new.append(f if tb is None else tb(f))
            feat_list = feat_new

        deep_feat = feat_list[-1]
        outs = [head(f) for head,f in zip(self.heads, feat_list)]
        return (outs, deep_feat) if return_deep_feature else outs  # list of (reg,obj,cls)

    # ONNX용: 외부 상태 입출력 (last scale 전제)
    def forward_external(self, x, state_in: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]):
        """state_in: None | [h](GRU) | [(h,c)](LSTM). temporal_on_scales='last' 전제."""
        assert (self.temporal is None) or (self.temporal_on_scales == "last"), \
            "ONNX 외부 상태 경로는 현재 temporal_on_scales='last'만 지원"

        x_input = x  # 원본 입력 보존

        # backbone/neck
        feats_memory = []
        for m in self.backbone_neck:
            if m.f != -1:
                x = feats_memory[m.f] if isinstance(m.f,int) else [x if j==-1 else feats_memory[j] for j in m.f]
            x = m(x)
            feats_memory.append(x if m.i in self.save_idx else None)
        feat_list = [feats_memory[i] for i in self.f_indices]
        if self.include_p2:
            p3 = feat_list[0]
            p2_src = feats_memory[self.p2_source_idx]
            p3_red = self.p2_reduce(p3)
            p2_lat = self.p2_lateral(p2_src)
            p2_up = self.p2_upsample(p3_red)
            p2 = torch.cat([p2_up, p2_lat], dim=1)
            p2 = self.p2_fuse(p2)
            feat_list = [p2] + feat_list

        state_out = None
        # temporal
        if self.temporal is not None and state_in is not None:
            feat_new = []
            last = len(feat_list) - 1
            for idx, f in enumerate(feat_list):
                if idx != last or self.temporal[-1] is None:
                    feat_new.append(f); continue
                y, s = self.temporal[-1].step_with_state(f, state_in[0])
                feat_new.append(y)
                state_out = [s]
            feat_list = feat_new
        elif self.temporal is not None and state_in is None:
            # 내부상태 경로와 동일 처리
            return self.forward(x_input, use_temporal=True), None

        outs = [head(f) for head,f in zip(self.heads, feat_list)]
        return outs, state_out

# Loss
def chamfer_2pts(pred_2x2, gt_2x2):
    # FP16 cdist overflow 방지
    pred_2x2 = pred_2x2.to(torch.float32)
    gt_2x2   = gt_2x2.to(torch.float32)
    d = torch.cdist(pred_2x2, gt_2x2, p=2)
    d1 = d.min(dim=-1).values.sum(dim=-1)
    d2 = d.min(dim=-2).values.sum(dim=-1)
    return (d1 + d2).mean()

class Strict2_5DLoss(nn.Module):
    def __init__(self, num_classes=1, eta_px=3.0, obj_pos_weight=1.2, lambda_cd=1.0, k_pos_cap=64,
                 cls_weights=None, cls_focal_gamma=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.eta_px = float(eta_px)
        self.lambda_cd = float(lambda_cd)
        self.k_pos_cap = int(k_pos_cap)  # 기본 64로 낮춰 과부하/timeout 완화
        self.mse = nn.MSELoss(reduction='sum')
        self.register_buffer("obj_pos_weight", torch.tensor([obj_pos_weight], dtype=torch.float32))
        if cls_weights is not None and len(cls_weights) != num_classes:
            print(f"[Warn] cls_weights length ({len(cls_weights)}) != num_classes ({num_classes}), ignoring weights")
            cls_weights = None
        self.register_buffer("cls_weights", torch.tensor(cls_weights, dtype=torch.float32) if cls_weights is not None else None)
        self.cls_focal_gamma = float(cls_focal_gamma)
        self.lambda_p0 = 1.0
        self._grid_cache = {}
    def set_p0_weight(self, w: float):
        self.lambda_p0 = float(w)

    @staticmethod
    def _grid_centers_cpu(Hs, Ws, stride):
        ys = (torch.arange(Hs, dtype=torch.float32) + 0.5) * float(stride)
        xs = (torch.arange(Ws, dtype=torch.float32) + 0.5) * float(stride)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        # (Hs, Ws, 2) on CPU (float32 고정)
        return torch.stack([xx, yy], dim=-1)

    def _grid_centers_cached(self, Hs, Ws, stride, device):
        key = (device, Hs, Ws, float(stride))
        if key not in self._grid_cache:
            ys = (torch.arange(Hs, device=device, dtype=torch.float32) + 0.5) * float(stride)
            xs = (torch.arange(Ws, device=device, dtype=torch.float32) + 0.5) * float(stride)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self._grid_cache[key] = torch.stack([xx, yy], dim=-1)
        return self._grid_cache[key]

    def forward(self, outputs, targets, strides):
        dev = outputs[0][0].device
        cpu = torch.device("cpu")

        L = len(outputs)
        B = outputs[0][0].shape[0]

        reg_loss = torch.tensor(0.0, device=dev)
        obj_loss = torch.tensor(0.0, device=dev)
        cls_loss = torch.tensor(0.0, device=dev)
        reg_p0_sum  = torch.tensor(0.0, device=dev)
        reg_p12_sum = torch.tensor(0.0, device=dev)
        pos_count = 0
        neg_count = 0

        for b in range(B):
            gt = targets[b]['points'].to(dev, dtype=torch.float32)   # (Ng,3,2) GPU FP32
            gt_lbl = targets[b].get("labels")
            if gt_lbl is not None:
                gt_lbl = gt_lbl.to(dev, dtype=torch.long)
            Ng = gt.shape[0]

            for l in range(L):
                pred_reg, pred_obj, pred_cls = outputs[l]
                pred_reg_i = pred_reg[b]                           # (6,Hs,Ws)
                pred_obj_i = pred_obj[b]                           # (1,Hs,Ws)
                pred_cls_i = pred_cls[b]                           # (C,Hs,Ws)
                Hs, Ws = pred_obj_i.shape[1:]
                stride = float(strides[l]) if not isinstance(strides[l], torch.Tensor) else float(strides[l].item())

                # 격자 캐시 (GPU)
                centers = self._grid_centers_cached(Hs, Ws, stride, dev)  # (Hs,Ws,2) GPU float32
                centers_flat = centers.view(-1, 2)                        # (P,2)

                # GPU 타깃 텐서
                obj_t = torch.zeros_like(pred_obj_i, device=dev)        # (1,Hs,Ws)

                # 회귀용 프리컴퓨트(GPU): (Hs,Ws,3,2) -> FP32로 변환
                pred_off_full = pred_reg_i.permute(1,2,0).contiguous().view(Hs, Ws, 3, 2).to(torch.float32)
                # 오프셋 클램프 (-64~64)
                pred_off_full = torch.clamp(pred_off_full, -64.0, 64.0)

                cls_pos_logits_gpu = [] if pred_cls_i.shape[0] == 1 else None
                cls_targets_all = None

                if Ng == 0:
                    obj_loss += F.binary_cross_entropy_with_logits(
                        pred_obj_i, obj_t, pos_weight=self.obj_pos_weight, reduction='sum'
                    )
                    neg_count += Hs*Ws
                    continue

                # GPU에서 GT 처리 → (iy, ix) 수집
                px = centers_flat[:, 0].view(1, -1)
                py = centers_flat[:, 1].view(1, -1)
                A = gt[:, 0, :]; Bp = gt[:, 1, :]; C = gt[:, 2, :]

                def _sign(Px, Py, Ax, Ay, Bx, By):
                    return (Px - Bx) * (Ay - By) - (Ax - Bx) * (Py - By)

                d1 = _sign(px, py, A[:, 0:1], A[:, 1:2], Bp[:, 0:1], Bp[:, 1:2])
                d2 = _sign(px, py, Bp[:, 0:1], Bp[:, 1:2], C[:, 0:1], C[:, 1:2])
                d3 = _sign(px, py, C[:, 0:1], C[:, 1:2], A[:, 0:1], A[:, 1:2])
                inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))  # (Ng,P)

                px_expand = px.expand(Ng, -1)
                py_expand = py.expand(Ng, -1)
                d_seg1 = _point_to_segment_dist(px_expand, py_expand,
                                                A[:, 0:1], A[:, 1:2], Bp[:, 0:1], Bp[:, 1:2])
                d_seg2 = _point_to_segment_dist(px_expand, py_expand,
                                                Bp[:, 0:1], Bp[:, 1:2], C[:, 0:1], C[:, 1:2])
                d_seg3 = _point_to_segment_dist(px_expand, py_expand,
                                                C[:, 0:1], C[:, 1:2], A[:, 0:1], A[:, 1:2])
                dist_b = torch.minimum(d_seg1, torch.minimum(d_seg2, d_seg3))  # (Ng,P)
                near = dist_b <= self.eta_px
                pos_mask_all = (inside | near)  # (Ng,P)

                pos_list_gpu = []   # [(Np,2) GPU] accumulate
                tri_list_gpu = []   # 매칭되는 GT 삼각형(GPU) (회귀에 필요)
                cls_list = []
                for j in range(Ng):
                    if gt_lbl is not None and gt_lbl.shape[0] > j:
                        cls_idx = int(gt_lbl[j].item())
                    else:
                        cls_idx = 0

                    mask_flat = pos_mask_all[j]
                    npix = int(mask_flat.sum().item())
                    if npix == 0:
                        continue

                    if npix > self.k_pos_cap:
                        flat_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)  # (Np,)
                        d_keep = dist_b[j][mask_flat]                                    # (Np,)
                        topk = torch.topk(-d_keep, k=self.k_pos_cap).indices
                        sel_flat = flat_idx[topk]
                    else:
                        sel_flat = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)

                    iy = torch.div(sel_flat, Ws, rounding_mode='floor')
                    ix = sel_flat - iy * Ws
                    pos_list_gpu.append(torch.stack([iy, ix], dim=1))  # (N,2) GPU
                    tri_list_gpu.append(gt[j])  # (3,2) on GPU, 같은 순서로 저장
                    cls_list.append(cls_idx)

                if len(pos_list_gpu) == 0:
                    # pos 없음
                    obj_loss += F.binary_cross_entropy_with_logits(
                        pred_obj_i, obj_t, pos_weight=self.obj_pos_weight, reduction='sum'
                    )
                    neg_count += Hs*Ws
                    continue

                # GPU로 모으기
                pos_idx = torch.cat(pos_list_gpu, dim=0)                  # (Npos,2) GPU

                # 평탄 인덱스 (GPU)
                flat_idx_gpu = (pos_idx[:,0] * Ws + pos_idx[:,1]).long()
                obj_t.view(-1).scatter_(0, flat_idx_gpu, 1.0)

                # 회귀 손실 (GPU, FP32)
                ix = pos_idx[:,1].float()
                iy = pos_idx[:,0].float()
                anchor_xy = torch.stack(((ix + 0.5) * stride, (iy + 0.5) * stride), dim=1).to(torch.float32)

                # 블록 길이 계산 및 클래스 반복
                lens = [p.shape[0] for p in pos_list_gpu]
                cls_targets = []
                tri_expanded = []
                for tri, ln in zip(tri_list_gpu, lens):
                    tri_expanded.append(tri.unsqueeze(0).expand(ln, -1, -1))
                for cls_idx, ln in zip(cls_list, lens):
                    if ln <= 0:
                        continue
                    cls_targets.append(torch.full((ln,), cls_idx, dtype=torch.long, device=dev))
                tri_rep = torch.cat(tri_expanded, dim=0).to(torch.float32)      # (Npos,3,2) GPU FP32
                if len(cls_targets) > 0:
                    cls_targets_all = torch.cat(cls_targets, dim=0)

                anchor_rep = anchor_xy[:,None,:].expand(-1,3,-1)                 # (Npos,3,2)
                gt_off_cells = (tri_rep - anchor_rep) / float(stride)            # (Npos,3,2) FP32

                pred_off_pos = pred_off_full[pos_idx[:,0], pos_idx[:,1]]         # (Npos,3,2) FP32

                # 회귀 손실을 FP32로 계산
                loss_p0 = self.mse(pred_off_pos[:,0,:], gt_off_cells[:,0,:])
                loss_cd = chamfer_2pts(pred_off_pos[:,1:,:], gt_off_cells[:,1:,:]) * pred_off_pos.shape[0]

                reg_loss   += self.lambda_p0 * loss_p0 + self.lambda_cd * loss_cd
                reg_p0_sum += self.lambda_p0 * loss_p0
                reg_p12_sum+= self.lambda_cd * loss_cd
                pos_count  += int(pos_idx.shape[0])

                # obj/cls BCE
                obj_loss += F.binary_cross_entropy_with_logits(
                    pred_obj_i, obj_t, pos_weight=self.obj_pos_weight, reduction='sum'
                )
                if pred_cls_i.shape[0] == 1:
                    if cls_pos_logits_gpu is not None and pos_idx.shape[0] > 0:
                        cls_pos_logits_gpu.append(pred_cls_i[0][pos_idx[:,0], pos_idx[:,1]])
                    if cls_pos_logits_gpu and len(cls_pos_logits_gpu) > 0:
                        logits = torch.cat(cls_pos_logits_gpu, dim=0)
                        cls_loss += F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='sum')
                else:
                    if cls_targets_all is not None and cls_targets_all.numel() > 0:
                        logits = pred_cls_i[:, pos_idx[:,0], pos_idx[:,1]].permute(1,0).contiguous()
                        weight = None
                        if self.cls_weights is not None:
                            weight = self.cls_weights.to(dev)
                        if self.cls_focal_gamma > 0.0:
                            log_probs = F.log_softmax(logits, dim=1)
                            probs = log_probs.exp()
                            ce = F.nll_loss(log_probs, cls_targets_all, weight=weight, reduction='none')
                            pt = probs[torch.arange(cls_targets_all.numel(), device=dev), cls_targets_all]
                            focal = ((1.0 - pt) ** self.cls_focal_gamma) * ce
                            cls_loss += focal.sum()
                        else:
                            cls_loss += F.cross_entropy(logits, cls_targets_all, weight=weight, reduction='sum')

                pos_now = int((obj_t>0.5).sum().item())
                neg_count += (Hs*Ws - pos_now)

        pos_eps = max(1, pos_count)
        reg_loss = reg_loss / pos_eps
        obj_loss = obj_loss / (pos_eps + max(1,neg_count))
        cls_loss = cls_loss / pos_eps

        reg_p0  = reg_p0_sum  / pos_eps
        reg_p12 = reg_p12_sum / pos_eps

        total = reg_loss + obj_loss + cls_loss

        # NaN/Inf 방어
        if not torch.isfinite(total):
            total = torch.tensor(0.0, device=dev)

        logs = {"loss_total": float(total.item()),
                "loss_reg": float(reg_loss.item()),
                "loss_obj": float(obj_loss.item()),
                "loss_cls": float(cls_loss.item()),
                "loss_p0": float(reg_p0.item()),
                "loss_p12": float(reg_p12.item()),
                "pos": pos_count, "neg": neg_count}
        return total, logs

# Save/Resume
def save_ckpt(path, epoch, model, opt_bb, opt_hd, sched_bb, sched_hd, scaler, best_val, extra: dict = None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt_bb": opt_bb.state_dict(),
        "opt_hd": opt_hd.state_dict(),
        "sched_bb": sched_bb.state_dict(),
        "sched_hd": sched_hd.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val,
    }
    if extra: ckpt.update(extra)
    torch.save(ckpt, path)

def load_ckpt(path, device, model, opt_bb, opt_hd, sched_bb, sched_hd, scaler):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    opt_bb.load_state_dict(ckpt["opt_bb"]); opt_hd.load_state_dict(ckpt["opt_hd"])
    if "sched_bb" in ckpt: sched_bb.load_state_dict(ckpt["sched_bb"])
    if "sched_hd" in ckpt: sched_hd.load_state_dict(ckpt["sched_hd"])
    if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    best = ckpt.get("best_val", -1.0)
    return epoch, best

# grad 유무 체크
def _opt_has_grad(opt):
    for g in opt.param_groups:
        for p in g["params"]:
            if p.grad is not None:
                return True
    return False

def _sanitize_grads(module: nn.Module):
    # 비정상 grad를 0으로 치환
    for p in module.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

# ONNX Export Helpers
class _ONNXWrapNone(nn.Module):
    def __init__(self, core: YOLO11_2_5D):
        super().__init__()
        self.core = core
    def forward(self, x):
        outs = self.core.forward(x, use_temporal=False)  # temporal none or 무시
        # list-of-3-tuples -> flat tuple
        return tuple([t for out in outs for t in out])

class _ONNXWrapGRULast(nn.Module):
    def __init__(self, core: YOLO11_2_5D):
        super().__init__()
        self.core = core
    def forward(self, x, h_in):
        outs, s_out = self.core.forward_external(x, state_in=[h_in])
        h_out = s_out[0]
        return tuple([t for out in outs for t in out] + [h_out])

class _ONNXWrapLSTMLast(nn.Module):
    def __init__(self, core: YOLO11_2_5D):
        super().__init__()
        self.core = core
    def forward(self, x, h_in, c_in):
        outs, s_out = self.core.forward_external(x, state_in=[(h_in, c_in)])
        h_out, c_out = s_out[0]
        return tuple([t for out in outs for t in out] + [h_out, c_out])

def export_epoch_onnx(model: YOLO11_2_5D, onnx_dir: Path, onnx_name: str,
                      img_size: Tuple[int,int], temporal: str, temporal_hidden: int):
    onnx_dir.mkdir(parents=True, exist_ok=True)
    H, W = img_size
    device = next(model.parameters()).device
    model.eval()
    num_heads = len(model.heads)

    last_stride = int(model.strides[-1]) if isinstance(model.strides[-1], (int,float)) else int(model.strides[-1].item())
    Hh, Wh = H // last_stride, W // last_stride

    x = torch.zeros(1,3,H,W, device=device)

    # 래퍼/입력/동적축
    if temporal == "none" or model.temporal is None:
        wrapper = _ONNXWrapNone(model).to(device)
        inputs = (x,)
        input_names = ["images"]
        output_names = [f"p{l}_{k}" for l in range(num_heads) for k in ("reg","obj","cls")]
        dynamic_axes = { "images": {0: "batch"} }
    elif temporal == "gru":
        h = torch.zeros(1, temporal_hidden, Hh, Wh, device=device)
        wrapper = _ONNXWrapGRULast(model).to(device)
        inputs = (x, h)
        input_names = ["images", "h_in"]
        output_names = [f"p{l}_{k}" for l in range(num_heads) for k in ("reg","obj","cls")] + ["h_out"]
        dynamic_axes = { "images": {0: "batch"}, "h_in": {0: "batch"}, "h_out": {0: "batch"} }
    elif temporal == "lstm":
        h = torch.zeros(1, temporal_hidden, Hh, Wh, device=device)
        c = torch.zeros(1, temporal_hidden, Hh, Wh, device=device)
        wrapper = _ONNXWrapLSTMLast(model).to(device)
        inputs = (x, h, c)
        input_names = ["images", "h_in", "c_in"]
        output_names = [f"p{l}_{k}" for l in range(num_heads) for k in ("reg","obj","cls")] + ["h_out","c_out"]
        dynamic_axes = { "images": {0: "batch"}, "h_in": {0: "batch"}, "c_in": {0: "batch"},
                         "h_out": {0: "batch"}, "c_out": {0: "batch"} }
    else:
        raise NotImplementedError("temporal_on_scales='all' ONNX export는 아직 미지원")

    out_path = onnx_dir / onnx_name
    torch.onnx.export(
        wrapper, inputs, str(out_path),
        input_names=input_names, output_names=output_names,
        opset_version=17, do_constant_folding=True,
        dynamic_axes=dynamic_axes
    )
    print(f"[ONNX] saved: {out_path}")

# Train
def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 2.5D (paper-accurate, cls@pos-only, fast val)")
    parser.add_argument("--train-root", type=str, default="./output_all")
    parser.add_argument("--val-root", type=str, default=None)
    parser.add_argument("--data-layout", type=str, default="auto",
                        choices=["auto","A","B"],
                        help="A: <root>/images & <root>/labels, B: <root>/<vid>/{images,labels}, auto: 자동판별")

    parser.add_argument("--val-mode", choices=["none", "loss", "metrics"], default="metrics")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-batch", type=int, default=1)
    parser.add_argument("--val-max-batches", type=int, default=None)

    # 평가 하이퍼
    parser.add_argument("--eval-conf", type=float, default=0.60)
    parser.add_argument("--eval-nms-iou", type=float, default=0.20)
    parser.add_argument("--eval-topk", type=int, default=100)
    parser.add_argument("--eval-iou-thr", type=float, default=0.5)
    parser.add_argument("--score-mode", type=str, default="obj*cls", choices=["cls","obj","obj*cls"])
    parser.add_argument("--eval-class-conf", type=str, default=None,
                        help="클래스별 conf, 예: '0:0.2,1:0.35,2:0.35'")
    parser.add_argument("--clip-cells", type=float, default=4.0)

    # 라벨/손실 하이퍼
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--eta-px", type=float, default=3.0)
    parser.add_argument("--lambda-cd", type=float, default=1.0)
    parser.add_argument("--k-pos-cap", type=int, default=96)
    parser.add_argument("--cls-weights", type=float, nargs="+", default=None,
                        help="클래스별 가중치(CE). num-classes 길이와 맞아야 함.")
    parser.add_argument("--cls-focal-gamma", type=float, default=0.0,
                        help=">0이면 focal loss로 적용 (gamma 값)")

    # 백본 프리즈
    parser.add_argument("--freeze-bb-epochs", type=int, default=1)

    # 가중치 관련
    parser.add_argument("--yolo-weights", type=str, default="yolo11m.pt")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-epoch", type=int, default=0)

    # 공통 하이퍼
    parser.add_argument("--img-h", type=int, default=864)
    parser.add_argument("--img-w", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lr-bb", type=float, default=2e-4)
    parser.add_argument("--lr-hd", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-4)

    # 스케일 옵션
    parser.add_argument("--p2-head", action="store_true", default=False,
                        help="stride=4 (P2) 헤드를 추가합니다. 기존 가중치는 호환되지 않으며 새로 학습해야 합니다.")

    # 안정화 옵션
    parser.add_argument("--skip-bad-batch", action="store_true",
                        help="손실/그라디언트가 NaN/Inf이면 해당 배치 스킵")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    # 저장 경로 & CSV
    parser.add_argument("--save-dir", type=str, default="./results/train")
    parser.add_argument("--log-csv", type=str, default=None)  # None이면 save-dir 내부에 자동 생성
    parser.add_argument("--train-augment", action="store_true", default=True,
                        help="Albumentations 기반 학습 증강 사용")
    parser.add_argument("--no-train-augment", action="store_false", dest="train_augment",
                        help="증강 비활성화")

    # Temporal 옵션
    parser.add_argument("--temporal", type=str, default="lstm",
                        choices=["none", "gru", "lstm"],
                        help="헤드 앞 시계열 모듈 선택 (none|gru|lstm)")
    parser.add_argument("--temporal-hidden", type=int, default=256,
                        help="ConvGRU/LSTM hidden 채널 수")
    parser.add_argument("--temporal-layers", type=int, default=1,
                        help="스택 층 수(1~2 권장)")
    parser.add_argument("--temporal-on-scales", type=str, default="last",
                        choices=["last", "all"],
                        help="'last'는 가장 깊은 스케일만 적용, 'all'은 모든 스케일 적용")
    parser.add_argument("--temporal-reset-per-batch", action="store_true",
                        help="학습 시 배치마다 상태 리셋(셔플 학습 권장). 검증/실시간 스트림은 끄는 게 좋음.")

    # Sequence 옵션
    parser.add_argument("--seq-len", type=int, default=4, help="시퀀스 길이 T (ConvRNN 학습은 T>=4 권장)")
    parser.add_argument("--seq-stride", type=int, default=1, help="윈도우 간 간격")
    parser.add_argument("--seq-grouping", type=str, default="auto",
                        choices=["auto","by_subdir","by_prefix","flat"],
                        help="영상 단위 그룹핑 방식: 하위폴더, 파일명 prefix, 또는 전부 하나(flat)")
    parser.add_argument("--tbptt-detach", action="store_true",
                        help="시퀀스 내부 time-step마다 hidden detach(폭주/메모리 방지)")
    parser.add_argument("--seq-streaming", action="store_true", default=True,
                        help="윈도우를 섞지 않고 입력 순서를 유지하며 동일 vid에서는 hidden state를 이어감")
    parser.add_argument("--no-seq-streaming", action="store_false", dest="seq_streaming",
                        help="기존 방식처럼 윈도우 shuffle + 배치 단위 state 초기화")
    # DSI / GAM 옵션
    parser.add_argument("--dsi", action="store_true", default=True,
                        help="Vision-Teacher 기반 Deep Semantic Injection 사용")
    parser.add_argument("--no-dsi", action="store_false", dest="dsi",
                        help="DSI 비활성화")
    parser.add_argument("--dsi-teacher-arch", type=str, default="vit_b_16",
                        choices=["vit_b_16","vit_l_16"])
    parser.add_argument("--dsi-teacher-checkpoint", type=str, default=None)
    parser.add_argument("--dsi-teacher-resize", type=int, nargs=2, metavar=("H","W"), default=(384,384))
    parser.add_argument("--dsi-weight", type=float, default=5.0)
    parser.add_argument("--dsi-weight-min", type=float, default=1.0)
    parser.add_argument("--dsi-weight-max", type=float, default=30.0)
    parser.add_argument("--dsi-warmup-epochs", type=int, default=5)
    parser.add_argument("--dsi-mean", type=float, nargs=3, default=(0.485,0.456,0.406))
    parser.add_argument("--dsi-std", type=float, nargs=3, default=(0.229,0.224,0.225))
    parser.add_argument("--dsi-pretrained-teacher", action="store_true", default=True)
    parser.add_argument("--dsi-no-pretrained-teacher", action="store_false",
                        dest="dsi_pretrained_teacher")
    parser.add_argument("--dsi-gam", action="store_true", default=True,
                        help="GAM 기반 dsi weight 자동 조절")
    parser.add_argument("--no-dsi-gam", action="store_false", dest="dsi_gam",
                        help="DSI GAM 비활성화")
    parser.add_argument("--dsi-gam-rho", type=float, default=0.4)
    parser.add_argument("--dsi-gam-delta", type=float, default=0.05)

    args = parser.parse_args()

    # 모델 이름 stem
    model_stem = Path(args.yolo_weights).stem
    # 저장 경로 구성
    save_dir = Path(args.save_dir).expanduser().resolve()
    pth_dir = save_dir / "pth"
    ckpt_dir = save_dir / "ckpt"
    onnx_dir = save_dir / "onnx"
    pth_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # CSV 경로 결정
    csv_path = Path(args.log_csv) if args.log_csv else (save_dir / f"{model_stem}_train_log.csv")
    need_header = (not csv_path.exists())

    IMG_SIZE = (args.img_h, args.img_w)
    TRAIN_BATCH = args.batch
    EPOCHS = args.epochs
    LR_MIN = args.lr_min
    WD = args.wd
    NUM_CLASSES = int(args.num_classes)

    eval_cfg = None
    if args.val_mode == "metrics":
        class_conf_map = None
        if args.eval_class_conf:
            class_conf_map = {}
            for tok in args.eval_class_conf.split(","):
                tok = tok.strip()
                if not tok: continue
                try:
                    cid, thr = tok.split(":")
                    class_conf_map[int(cid.strip())] = float(thr.strip())
                except Exception:
                    print(f"[Warn] invalid --eval-class-conf token '{tok}', expected 'cls:thr'")
        eval_cfg = {
            "conf_th": args.eval_conf,
            "nms_iou": args.eval_nms_iou,
            "topk": args.eval_topk,
            "iou_thr": args.eval_iou_thr,
            "score_mode": args.score_mode,
            "clip_cells": args.clip_cells,
            "class_conf": class_conf_map,
        }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset / DataLoader
    train_transform = None
    if args.train_augment:
        train_transform = build_default_train_augment(IMG_SIZE)
    train_dataset = ParallelogramDataset(
        args.train_root,
        target_size=IMG_SIZE,
        transform=train_transform,
        data_layout=args.data_layout
    )
    if len(train_dataset) == 0:
        print("No training data. Exit."); return

    if args.seq_len >= 2:
        train_seq = SeqWindowDataset(train_dataset,
                                     grouping=args.seq_grouping,
                                     seq_len=args.seq_len, seq_stride=args.seq_stride)
        train_loader = DataLoader(train_seq, batch_size=TRAIN_BATCH,
                                  shuffle=(not args.seq_streaming),
                                  collate_fn=collate_seq_fn, num_workers=4, pin_memory=True,
                                  persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH,
                                  shuffle=(not args.seq_streaming),
                                  collate_fn=collate_fn, num_workers=4, pin_memory=True,
                                  persistent_workers=True)

    val_loader = None
    if args.val_mode != "none":
        val_root = args.val_root or args.train_root
        val_dataset = ParallelogramDataset(
            val_root,
            target_size=IMG_SIZE,
            transform=None,
            data_layout=args.data_layout
        )
        if len(val_dataset) == 0:
            print("No validation data. Exit."); return
        if args.seq_len >= 2:
            val_seq = SeqWindowDataset(val_dataset,
                                       grouping=args.seq_grouping,
                                       seq_len=args.seq_len, seq_stride=args.seq_stride)
            val_loader = DataLoader(val_seq, batch_size=args.val_batch, shuffle=False,
                                    collate_fn=collate_seq_fn, num_workers=4, pin_memory=True,
                                    persistent_workers=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False,
                                    collate_fn=collate_fn, num_workers=4, pin_memory=True,
                                    persistent_workers=True)

    # Model
    model = YOLO11_2_5D(
        yolo11_path=args.yolo_weights,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        temporal_mode=args.temporal,
        temporal_hidden=args.temporal_hidden,
        temporal_layers=args.temporal_layers,
        temporal_on_scales=args.temporal_on_scales,
        include_p2=args.p2_head
    ).to(DEVICE)

    if isinstance(model.strides, torch.Tensor):
        model.strides = [float(s.item()) for s in model.strides]
    else:
        model.strides = [float(s) for s in model.strides]

    dsi_helper = None
    dsi_weight = float(args.dsi_weight)
    teacher_resize = tuple(args.dsi_teacher_resize) if args.dsi_teacher_resize else None
    if args.dsi:
        dsi_helper = DeepSemanticInjector(
            student_channels=model.deep_feature_channels,
            arch=args.dsi_teacher_arch,
            pretrained=(args.dsi_teacher_checkpoint is None) and args.dsi_pretrained_teacher,
            checkpoint=args.dsi_teacher_checkpoint,
            resize_hw=teacher_resize,
            mean=tuple(args.dsi_mean),
            std=tuple(args.dsi_std),
            device=DEVICE
        ).to(DEVICE)
        print(f"[DSI] Enabled (teacher={args.dsi_teacher_arch}, resize={teacher_resize}, weight={dsi_weight})")

    criterion = Strict2_5DLoss(num_classes=NUM_CLASSES,
                               eta_px=args.eta_px,
                               obj_pos_weight=1.2,
                               lambda_cd=args.lambda_cd,
                               k_pos_cap=args.k_pos_cap,
                               cls_weights=args.cls_weights,
                               cls_focal_gamma=args.cls_focal_gamma).to(DEVICE)

    head_params = list(model.heads.parameters())
    bb_params = [p for n,p in model.named_parameters() if not n.startswith("heads.") and "temporal" not in n]
    temporal_params = []
    for n, p in model.named_parameters():
        if "temporal" in n:
            temporal_params.append(p)

    opt_bb = optim.SGD(bb_params, lr=args.lr_bb, momentum=0.9, weight_decay=WD, nesterov=True)
    opt_hd = optim.AdamW(list(head_params) + temporal_params, lr=args.lr_hd, betas=(0.9, 0.999), weight_decay=WD)
    if dsi_helper is not None:
        opt_hd.add_param_group({"params": dsi_helper.projector.parameters(), "lr": args.lr_hd})

    sched_bb = optim.lr_scheduler.CosineAnnealingLR(opt_bb, T_max=EPOCHS, eta_min=LR_MIN)
    sched_hd = optim.lr_scheduler.CosineAnnealingLR(opt_hd, T_max=EPOCHS, eta_min=LR_MIN)

    scaler = GradScaler(enabled=(DEVICE == 'cuda'))

    def set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters(): p.requires_grad = flag

    # CSV 헤더
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow([
                "epoch","train_loss","val_loss",
                "precision","recall","map50","mAOE_deg",
                "lr_bb","lr_hd","best_map"
            ])

    best_val = -1.0 if args.val_mode == "metrics" else float('inf')
    start_ep = 0
    if args.resume is not None:
        start_ep, best_val = load_ckpt(args.resume, DEVICE, model, opt_bb, opt_hd, sched_bb, sched_hd, scaler)
        print(f"[Resume] {args.resume} -> start_ep={start_ep}, best_val={best_val:.4f}")
    elif args.weights is not None:
        sd = torch.load(args.weights, map_location=DEVICE)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]

        # cls 수 불일치 시 헤드 제외
        ckpt_classes = None
        for k, v in sd.items():
            if k.startswith("heads.") and ".cls." in k and hasattr(v, "shape") and len(v.shape) > 0:
                ckpt_classes = v.shape[0]
                break
        if (ckpt_classes is not None) and (ckpt_classes != args.num_classes):
            print(f"[Warm-start] drop cls weights (ckpt_classes={ckpt_classes} != num_classes={args.num_classes})")
            sd = {k:v for k,v in sd.items() if not (k.startswith("heads.") and ".cls." in k)}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Warm-start] {args.weights} (missing={len(missing)}, unexpected={len(unexpected)})")
        start_ep = int(args.start_epoch)

    val_desc = f"{args.val_mode} (interval={args.val_interval})" if args.val_mode != "none" else "disabled"
    print(f"[Train] device={DEVICE}, epochs={EPOCHS}, start_ep={start_ep}, batch={TRAIN_BATCH}, strides={model.strides}, val={val_desc}")
    print(f"[Save] base='{model_stem}', save_dir='{save_dir}'")

    def run_epoch(loader, train=True, epoch_idx=0, eval_cfg=None, val_mode="metrics"):
        nonlocal dsi_weight
        warmup_ep = 5
        w = 0.5 + 0.5 * min(epoch_idx, warmup_ep-1) / (warmup_ep-1)
        criterion.set_p0_weight(w)

        if train:
            if epoch_idx < args.freeze_bb_epochs: set_requires_grad(model.backbone_neck, False)
            else: set_requires_grad(model.backbone_neck, True)

        model.train(mode=train)
        collect_metrics = (not train) and (eval_cfg is not None)
        metric_records = []; total_gt = 0
        total_gt_per_cls = defaultdict(int)

        tot = treg = tobj = tcls = tpos = tp0 = tp12 = 0.0
        nb = 0; batches_seen = 0
        total_loss_sum = 0.0
        dsi_loss_sum = 0.0
        use_dsi = train and (dsi_helper is not None)
        if args.dsi_warmup_epochs > 0:
            dsi_warm_factor = min(1.0, (epoch_idx + 1) / float(args.dsi_warmup_epochs))
        else:
            dsi_warm_factor = 1.0
        grad_ratio_sum = 0.0
        grad_ratio_count = 0
        prev_stream_key = None

        def handle_temporal_state(batch_keys):
            nonlocal prev_stream_key
            if args.temporal == "none":
                return
            if not args.seq_streaming:
                if train:
                    if args.temporal_reset_per_batch:
                        model.reset_temporal()
                    else:
                        model.detach_temporal()
                else:
                    model.reset_temporal()
                return

            key = None
            same_key = True
            if batch_keys:
                key = batch_keys[0]
                for k in batch_keys:
                    if k != key:
                        same_key = False
                        break

            if args.temporal_reset_per_batch:
                model.reset_temporal()
                prev_stream_key = key
                return

            if (not same_key) or key is None:
                model.reset_temporal()
                prev_stream_key = key
                return

            if prev_stream_key is None or key != prev_stream_key:
                model.reset_temporal()
            else:
                model.detach_temporal()
            prev_stream_key = key

        pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'} {epoch_idx+1}/{EPOCHS} (p0_w={w:.2f})")
        for batch in pbar:
            if args.seq_len >= 2:
                imgs_bt, tgts_bt, names_bt, vids_bt, T = batch   # imgs: (B,T,3,H,W)
                B = imgs_bt.size(0)

                handle_temporal_state(vids_bt)

                if train:
                    opt_bb.zero_grad(set_to_none=True); opt_hd.zero_grad(set_to_none=True)

                logs_accum = {"loss_total":0,"loss_reg":0,"loss_obj":0,"loss_cls":0,
                              "loss_p0":0,"loss_p12":0,"pos":0,"neg":0}

                bad_batch = False
                loss_seq = None
                steps_seq = 0
                loss_batch_sum = 0.0
                dsi_batch_sum = 0.0

                for tstep in range(T):
                    imgs = imgs_bt[:, tstep].to(DEVICE, non_blocking=True)
                    targets_cpu = [tgts_bt[b][tstep] for b in range(B)]
                    targets_dev = [{"points": t["points"].to(DEVICE), "labels": t["labels"].to(DEVICE)} for t in targets_cpu]

                    if train:
                        with amp_autocast():
                            if use_dsi:
                                outs, deep_feat = model(imgs, use_temporal=True, return_deep_feature=True)
                            else:
                                outs = model(imgs, use_temporal=True)
                                deep_feat = None
                            det_loss, logs = criterion(outs, targets_dev, model.strides)
                            dsi_loss = None
                            loss = det_loss
                            if use_dsi and deep_feat is not None:
                                dsi_loss = dsi_helper(deep_feat, imgs)
                                loss = loss + (dsi_weight * dsi_warm_factor) * dsi_loss
                        if not torch.isfinite(loss):
                            bad_batch = True
                            break
                        steps_seq += 1
                        if args.tbptt_detach and args.temporal != "none":
                            scaler.scale(loss / max(1, T)).backward()
                            model.detach_temporal()
                            if dsi_loss is not None:
                                dsi_batch_sum += float(dsi_loss.detach().item())
                        else:
                            loss_seq = loss if loss_seq is None else (loss_seq + loss)
                        loss_batch_sum += float(loss.detach().item())
                        if (not args.tbptt_detach) and dsi_loss is not None:
                            dsi_batch_sum += float(dsi_loss.detach().item())
                    else:
                        if val_mode == "loss":
                            with torch.inference_mode():
                                outs = model(imgs, use_temporal=True)
                                loss, logs = criterion(outs, targets_dev, model.strides)
                        else:
                            with torch.inference_mode(), amp_autocast():
                                outs = model(imgs, use_temporal=True)
                            loss = torch.tensor(0., device=imgs.device)
                            logs = {"loss_total":0,"loss_reg":0,"loss_obj":0,"loss_cls":0,"loss_p0":0,"loss_p12":0,"pos":0,"neg":0}
                        dsi_loss = None
                        loss_batch_sum += float(loss.detach().item())

                    for k in logs_accum.keys():
                        logs_accum[k] += logs.get(k, 0.0)

                    # 메트릭: 시퀀스 마지막 step만 평가
                    if collect_metrics and (tstep == T-1):
                        decoded = decode_predictions(
                            outs, model.strides,
                            clip_cells=eval_cfg.get("clip_cells", None),
                            conf_th=eval_cfg.get("conf_th", 0.30),
                            nms_iou=eval_cfg.get("nms_iou", 0.20),
                            topk=eval_cfg.get("topk", 100),
                            score_mode=eval_cfg.get("score_mode", "obj*cls"),
                            use_gpu_nms=True
                        )
                        class_conf_map = eval_cfg.get("class_conf", None)
                        if class_conf_map:
                            def _keep(d):
                                cid = int(d.get("class_id", d.get("cls", 0)))
                                thr = class_conf_map.get(cid, eval_cfg.get("conf_th", 0.30))
                                return float(d.get("score", 0.0)) >= thr
                            decoded = [[d for d in dets_img if _keep(d)] for dets_img in decoded]
                        decoded = [tiny_filter_on_dets(dets_img=di, min_area=20.0, min_edge=3.0) for di in decoded]
                        for dets_img, tgt_cpu in zip(decoded, [tgts_bt[b][tstep] for b in range(B)]):
                            gt_tri = tgt_cpu["points"].cpu().numpy()
                            gt_cls = tgt_cpu["labels"].cpu().numpy() if "labels" in tgt_cpu else None
                            records, _ = evaluate_single_image(
                                dets_img, gt_tri,
                                gt_classes=gt_cls,
                                iou_thr=eval_cfg.get("iou_thr", 0.5)
                            )
                            metric_records.extend(records); total_gt += gt_tri.shape[0]
                            if gt_cls is not None:
                                for c in gt_cls:
                                    total_gt_per_cls[int(c)] += 1

                total_loss_sum += loss_batch_sum / max(1, steps_seq)
                if use_dsi:
                    dsi_loss_sum += dsi_batch_sum / max(1, steps_seq)

                if train:
                    if bad_batch and args.skip_bad_batch:
                        # 배치 스킵
                        opt_bb.zero_grad(set_to_none=True); opt_hd.zero_grad(set_to_none=True)
                        if args.temporal != "none":
                            model.reset_temporal()
                            prev_stream_key = None
                        pbar.set_postfix_str("skip bad batch (NaN/Inf)")
                        continue

                    # GradScaler 스텝
                    if (not args.tbptt_detach) and loss_seq is not None and steps_seq > 0:
                        scaler.scale(loss_seq / max(1, steps_seq)).backward()
                    has_bb = _opt_has_grad(opt_bb)
                    has_hd = _opt_has_grad(opt_hd)

                    if has_bb: scaler.unscale_(opt_bb)
                    if has_hd: scaler.unscale_(opt_hd)

                    # 비정상 grad 제거 + norm clip
                    _sanitize_grads(model)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    if use_dsi and args.dsi_gam:
                        backbone_norm = module_grad_l1(model.backbone_neck)
                        head_norm = module_grad_l1(model.heads)
                        if model.temporal is not None:
                            head_norm += module_grad_l1(model.temporal)
                        if dsi_helper is not None:
                            head_norm += module_grad_l1(dsi_helper.projector)
                        denom = backbone_norm + head_norm
                        if denom > 0:
                            grad_ratio_sum += (backbone_norm / denom)
                            grad_ratio_count += 1

                    if has_bb: scaler.step(opt_bb)
                    if has_hd: scaler.step(opt_hd)
                    scaler.update()
                    # ---

                # 평균 로그로 표시
                logs_mean = {k:(v/ T if isinstance(v,(int,float)) else v) for k,v in logs_accum.items()}
                tot += logs_mean["loss_total"]; treg += logs_mean["loss_reg"]; tobj += logs_mean["loss_obj"]
                tcls += logs_mean["loss_cls"];  tp0  += logs_mean["loss_p0"];  tp12 += logs_mean["loss_p12"]
                tpos += logs_mean["pos"]; nb += 1

            else:
                # 단일 프레임 경로
                imgs, targets_cpu, names = batch
                handle_temporal_state(names)
                imgs = imgs.to(DEVICE, non_blocking=True)
                targets_dev = [{"points": t["points"].to(DEVICE), "labels": t["labels"].to(DEVICE)} for t in targets_cpu]

                if train:
                    opt_bb.zero_grad(set_to_none=True); opt_hd.zero_grad(set_to_none=True)
                    with amp_autocast():
                        if use_dsi:
                            outs, deep_feat = model(imgs, use_temporal=True, return_deep_feature=True)
                        else:
                            outs = model(imgs, use_temporal=True)
                            deep_feat = None
                        loss, logs = criterion(outs, targets_dev, model.strides)
                        dsi_loss = None
                        if use_dsi and deep_feat is not None:
                            dsi_loss = dsi_helper(deep_feat, imgs)
                            loss = loss + (dsi_weight * dsi_warm_factor) * dsi_loss
                    if torch.isfinite(loss):
                        scaler.scale(loss).backward()
                    elif args.skip_bad_batch:
                        pbar.set_postfix_str("skip bad batch (NaN/Inf)")
                        if args.temporal != "none":
                            model.reset_temporal()
                            prev_stream_key = None
                        continue

                    # GradScaler 스텝
                    has_bb = _opt_has_grad(opt_bb)
                    has_hd = _opt_has_grad(opt_hd)

                    if has_bb: scaler.unscale_(opt_bb)
                    if has_hd: scaler.unscale_(opt_hd)

                    _sanitize_grads(model)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    if use_dsi and args.dsi_gam:
                        backbone_norm = module_grad_l1(model.backbone_neck)
                        head_norm = module_grad_l1(model.heads)
                        if model.temporal is not None:
                            head_norm += module_grad_l1(model.temporal)
                        if dsi_helper is not None:
                            head_norm += module_grad_l1(dsi_helper.projector)
                        denom = backbone_norm + head_norm
                        if denom > 0:
                            grad_ratio_sum += (backbone_norm / denom)
                            grad_ratio_count += 1

                    if has_bb: scaler.step(opt_bb)
                    if has_hd: scaler.step(opt_hd)
                    scaler.update()
                    # ---

                else:
                    if val_mode == "loss":
                        with torch.inference_mode():
                            outs = model(imgs, use_temporal=True)
                        loss, logs = criterion(outs, targets_dev, model.strides)
                    else:
                        with torch.inference_mode(), amp_autocast():
                            outs = model(imgs, use_temporal=True)
                        loss = torch.tensor(0., device=imgs.device)
                        logs = {"loss_total":0,"loss_reg":0,"loss_obj":0,"loss_cls":0,"loss_p0":0,"loss_p12":0,"pos":0,"neg":0}
                    dsi_loss = None

                tot+=logs["loss_total"]; treg+=logs["loss_reg"]; tobj+=logs["loss_obj"]
                tcls+=logs["loss_cls"]; tp0+=logs["loss_p0"]; tp12+=logs["loss_p12"]; tpos+=logs["pos"]; nb+=1
                total_loss_sum += float(loss.detach().item())
                if use_dsi and dsi_loss is not None:
                    dsi_loss_sum += float(dsi_loss.detach().item())

            lr_bb = opt_bb.param_groups[0]['lr']; lr_hd = opt_hd.param_groups[0]['lr']
            pbar.set_postfix(
                loss_det=(tot/nb if nb>0 else 0),
                reg=(treg/nb if nb>0 else 0),
                obj=(tobj/nb if nb>0 else 0),
                p0=(tp0/nb if nb>0 else 0),
                p12=(tp12/nb if nb>0 else 0),
                cls=(tcls/nb if nb>0 else 0),
                pos=(tpos/nb if nb>0 else 0),
                loss_total=(total_loss_sum/nb if nb>0 else 0),
                dsi=(dsi_loss_sum/nb if (nb>0 and use_dsi) else 0),
                lr_bb=lr_bb,
                lr_hd=lr_hd
            )

            batches_seen += 1
            if (not train) and args.val_max_batches is not None and batches_seen >= args.val_max_batches:
                break

        avg_loss = float(total_loss_sum/nb) if nb>0 else 0.0
        metrics = compute_detection_metrics(metric_records, total_gt) if collect_metrics else None
        if collect_metrics:
            per_class_metrics = {}
            # 클래스별 records/GT 수 집계
            class_records = defaultdict(list)
            for rec in metric_records:
                if len(rec) >= 5:
                    class_records[int(rec[4])].append(rec)
            class_ids = set(class_records.keys()) | set(total_gt_per_cls.keys())
            for cls_id in sorted(class_ids):
                recs = class_records.get(cls_id, [])
                per_class_metrics[int(cls_id)] = compute_detection_metrics(
                    recs, total_gt_per_cls.get(cls_id, 0)
                )
            metrics["per_class"] = per_class_metrics
        grad_ratio_avg = (grad_ratio_sum / grad_ratio_count) if grad_ratio_count > 0 else None
        return avg_loss, metrics, grad_ratio_avg

    # Loop
    for ep in range(start_ep, EPOCHS):
        train_loss, _, train_ratio = run_epoch(train_loader, train=True, epoch_idx=ep, eval_cfg=eval_cfg, val_mode=args.val_mode)
        if args.dsi and args.dsi_gam and train_ratio is not None:
            target = args.dsi_gam_rho
            delta = args.dsi_gam_delta
            new_weight = dsi_weight
            if train_ratio > target + delta:
                new_weight = max(args.dsi_weight_min, dsi_weight * (target / (train_ratio + 1e-8)))
            elif train_ratio < target - delta:
                new_weight = min(args.dsi_weight_max, dsi_weight * (target / (train_ratio + 1e-8)))
            if abs(new_weight - dsi_weight) > 1e-6:
                print(f"[GAM] epoch {ep+1} ratio={train_ratio:.4f} -> dsi_weight {dsi_weight:.3f} -> {new_weight:.3f}")
                dsi_weight = new_weight

        run_val = (args.val_mode != "none" and val_loader is not None and
                   ((ep + 1) % max(1, args.val_interval) == 0 or ep == EPOCHS - 1))

        val_loss = None
        curr_map = curr_prec = curr_rec = curr_maoe = None

        if run_val:
            val_loss, val_metrics, _ = run_epoch(val_loader, train=False, epoch_idx=ep, eval_cfg=eval_cfg, val_mode=args.val_mode)
            if val_metrics is not None:
                curr_prec = float(val_metrics["precision"])
                curr_rec  = float(val_metrics["recall"])
                curr_map  = float(val_metrics["map50"])
                curr_maoe = float(val_metrics["mAOE_deg"])
                print("  -> metrics P={:.4f} R={:.4f} mAP@50={:.4f} mAOE(deg)={:.2f}".format(
                    curr_prec, curr_rec, curr_map, curr_maoe
                ))
                per_cls = val_metrics.get("per_class")
                if per_cls:
                    pcs = []
                    for cls_id in sorted(per_cls.keys()):
                        m = per_cls[cls_id]
                        pcs.append(
                            f"{cls_id}:P={m['precision']:.3f} R={m['recall']:.3f} mAP50={m['map50']:.3f} mAOE={m['mAOE_deg']:.2f}"
                        )
                    print("  -> per-class " + " | ".join(pcs))
                # mAP@50 기준 베스트
                if curr_map > best_val:
                    best_val = curr_map
                    best_pth = pth_dir / f"{model_stem}_2_5d_best.pth"
                    torch.save(model.state_dict(), best_pth)
                    print(f"  -> best (mAP@50) updated to {best_val:.4f}, saved: {best_pth}")
            else:
                if val_loss is not None and args.val_mode == "loss" and val_loss < best_val:
                    best_val = val_loss
                    best_pth = pth_dir / f"{model_stem}_2_5d_best.pth"
                    torch.save(model.state_dict(), best_pth)
                    print(f"  -> best (val_loss) updated {best_val:.4f}, saved: {best_pth}")

        # 스케줄러 스텝
        sched_bb.step(); sched_hd.step()

        # 에폭별 저장
        pth_path  = pth_dir  / f"{model_stem}_2_5d_epoch_{ep+1:03d}.pth"
        ckpt_path = ckpt_dir / f"{model_stem}_2_5d_epoch_{ep+1:03d}.ckpt"
        torch.save(model.state_dict(), pth_path)
        save_ckpt(
            ckpt_path, epoch=ep+1, model=model,
            opt_bb=opt_bb, opt_hd=opt_hd, sched_bb=sched_bb, sched_hd=sched_hd, scaler=scaler,
            best_val=best_val
        )
        print(f"  -> saved {pth_path} and {ckpt_path}")

        # ONNX 내보내기
        try:
            if args.temporal_on_scales != "last" and args.temporal != "none":
                print("[ONNX] Skip: temporal_on_scales='all' 내보내기는 아직 미지원")
            else:
                onnx_name = f"{model_stem}_2_5d_epoch_{ep+1:03d}.onnx"
                export_epoch_onnx(
                    model=model,
                    onnx_dir=onnx_dir,
                    onnx_name=onnx_name,
                    img_size=IMG_SIZE,
                    temporal=args.temporal,
                    temporal_hidden=args.temporal_hidden
                )
        except Exception as e:
            print(f"[ONNX] Export failed at epoch {ep+1}: {e}")

        # CSV 로깅
        lr_bb = opt_bb.param_groups[0]['lr']; lr_hd = opt_hd.param_groups[0]['lr']
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep+1,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if curr_prec is None else f"{curr_prec:.6f}",
                "" if curr_rec  is None else f"{curr_rec:.6f}",
                "" if curr_map  is None else f"{curr_map:.6f}",
                "" if curr_maoe is None else f"{curr_maoe:.6f}",
                f"{lr_bb:.6e}",
                f"{lr_hd:.6e}",
                f"{best_val:.6f}" if isinstance(best_val, float) else best_val
            ])

    print("Done.")

if __name__ == "__main__":
    main()