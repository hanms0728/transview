#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLY 포인트클라우드 위에 BEV 라벨 기반으로 차량 GLB 메쉬를 오버레이하는 3D 뷰어."""

import os
import glob
import time
import math
import warnings
import argparse
import numpy as np
import open3d as o3d
import threading

def load_labels_dir(label_dir: str):
    """라벨 디렉토리 로드 (9열/6열 자동 인식)"""
    files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    frames = []
    for f in files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr = np.loadtxt(f, ndmin=2)
            if arr.size == 0:
                frames.append((f, np.zeros((0, 9), dtype=np.float32)))
                continue
            if arr.shape[1] < 6:
                frames.append((f, np.zeros((0, 9), dtype=np.float32)))
                continue
            arr = arr.astype(np.float32)
            if arr.shape[1] >= 9:
                # [cls, cx, cy, cz, L, W, yaw, pitch, roll]만 사용
                arr = arr[:, :9]
            else:
                # 6열 구버전: [cls, cx, cy, L, W, yaw] → cz=0, pitch=0, roll=0 추가
                cls_cx_cy = arr[:, :3]
                L = arr[:, 3:4]
                W = arr[:, 4:5]
                yaw = arr[:, 5:6]
                zeros = np.zeros((arr.shape[0], 3), dtype=np.float32)  # cz, pitch, roll
                # 재조합: cls cx cy cz L W yaw pitch roll
                arr = np.concatenate([cls_cx_cy, zeros[:, :1], L, W, yaw, zeros[:, 1:]], axis=1)
            frames.append((f, arr))
        except Exception:
            frames.append((f, np.zeros((0, 9), dtype=np.float32)))
    return frames

def unitize_mesh(mesh: o3d.geometry.TriangleMesh):
    """메쉬 중심 정렬 + 최대변 1.0 정규화"""
    mesh = mesh.compute_vertex_normals()
    # GLB가 Y-up이면 Z-up으로 보정
    try:
        APPLY_Y_UP_TO_Z_UP = True  # 필요시 False로
    except NameError:
        APPLY_Y_UP_TO_Z_UP = True
    bb = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bb.get_extent(), dtype=np.float32)
    scale = 1.0 / max(1e-9, extent.max())
    mesh.translate(-bb.get_center())
    mesh.scale(scale, center=(0, 0, 0))
    # Y-up → Z-up 보정
    if APPLY_Y_UP_TO_Z_UP:
        Rx90 = mesh.get_rotation_matrix_from_axis_angle([math.radians(90.0), 0.0, 0.0])
        mesh.rotate(Rx90, center=(0, 0, 0))
    return mesh

def build_unit_to_world_T(length, width, yaw_deg, center_xyz,
                          pitch_deg: float = 0.0, roll_deg: float = 0.0,
                          up_scale_from_width=0.5):
    """유닛 메쉬 → 월드 좌표 변환 4x4 행렬 (Rz·Ry·Rx 순)"""
    sx = max(1e-4, float(length))
    sy = max(1e-4, float(width))
    sz = max(1e-4, float(width) * up_scale_from_width)

    # 스케일
    S = np.diag([sx, sy, sz, 1.0]).astype(np.float64)

    # 회전행렬들 (deg → rad)
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))

    cz, szn = np.cos(yaw),   np.sin(yaw)
    cp, sp  = np.cos(pitch), np.sin(pitch)
    cr, sr  = np.cos(roll),  np.sin(roll)

    Rz = np.array([[ cz,-szn, 0, 0],
                   [ szn, cz, 0, 0],
                   [  0,   0, 1, 0],
                   [  0,   0, 0, 1]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp, 0],
                   [  0, 1,  0, 0],
                   [-sp, 0, cp, 0],
                   [  0, 0,  0, 1]], dtype=np.float64)
    Rx = np.array([[ 1,  0,  0, 0],
                   [ 0, cr, -sr, 0],
                   [ 0, sr,  cr, 0],
                   [ 0,  0,   0, 1]], dtype=np.float64)

    R = (Rz @ Ry @ Rx)

    # 평행이동
    Tt = np.eye(4, dtype=np.float64)
    Tt[:3, 3] = np.asarray(center_xyz, dtype=np.float64)

    # 고정 모델 보정이 unitize에서 적용되므로 여기서는 Tt @ R @ S만 사용
    T_model = (Tt @ R @ S)
    return T_model

def build_kdtree_for_z(cloud: o3d.geometry.PointCloud):
    pts = np.asarray(cloud.points)
    if pts.shape[0] == 0:
        return None, None
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    return kdtree, pts

def estimate_z_from_cloud(xy: np.ndarray,
                          kdtree: o3d.geometry.KDTreeFlann,
                          xyz_pts: np.ndarray,
                          radius: float,
                          default_z: float = 0.0):
    """(x,y) 근방 포인트의 z 중앙값"""
    if xy.ndim == 1:
        xy = xy.reshape(1, 2)
    zs = []
    for i in range(xy.shape[0]):
        center = np.array([xy[i, 0], xy[i, 1], 0.0], dtype=np.float64)
        [_, idxs, _] = kdtree.search_hybrid_vector_3d(center, radius, 512)
        if len(idxs) == 0:
            zs.append(default_z)
        else:
            zvals = xyz_pts[idxs, 2]
            zs.append(float(np.median(zvals)))
    return np.array(zs, dtype=np.float32).reshape(-1)

def _flip_y_T():
    T = np.eye(4, dtype=np.float64)
    T[1, 1] = -1.0
    return T

def main():
    ap = argparse.ArgumentParser("Overlay GLB vehicles on global PLY with BEV labels (free-cam, optimized)")
    ap.add_argument("--global-ply", type=str, required=True, help="fused/global_fused.ply")
    ap.add_argument("--bev-label-dir", type=str, required=True,
                    help=".../bev_labels (정렬됨). 포맷: 9열 'cls cx cy cz L W yaw pitch roll' 또는 구버전 6열.")
    ap.add_argument("--vehicle-glb", type=str, required=True, help="차량 메쉬(.glb 권장; .obj도 가능)")

    ap.add_argument("--fps", type=float, default=30.0, help="재생 FPS")
    ap.add_argument("--play", action="store_true", help="자동 재생 시작 (기본은 일시정지)")
    ap.add_argument("--pause-on-empty", action="store_true", help="프레임에 디텍션 없으면 일시정지")
    ap.add_argument("--bg-dark", action="store_true", help="배경을 검정으로 설정 시도")
    ap.add_argument("--force-legacy", action="store_true", default=True,
                    help="레거시 Visualizer 사용 (기본: on)")
    ap.add_argument("--no-force-legacy", dest="force_legacy", action="store_false",
                    help="O3DVisualizer(modern) 사용 시도")
    # Y-axis inversion defaults: enabled; provide --no-invert-... to disable
    ap.add_argument("--invert-bev-y", dest="invert_bev_y", action="store_true",
                    help="BEV 좌표계의 Y축을 반전하여 시각화(cy=-cy, yaw=-yaw). (기본: ON)")
    ap.add_argument("--no-invert-bev-y", dest="invert_bev_y", action="store_false",
                    help="BEV Y축 반전을 비활성화합니다.")
    ap.set_defaults(invert_bev_y=True)

    ap.add_argument("--invert-ply-y", dest="invert_ply_y", action="store_true",
                    help="PLY 월드의 Y축을 반전(거울 뒤집기)하여 시각화합니다. (기본: ON, point cloud에만 적용)")
    ap.add_argument("--no-invert-ply-y", dest="invert_ply_y", action="store_false",
                    help="PLY Y축 반전을 비활성화합니다.")
    ap.set_defaults(invert_ply_y=True)

    # Z 추정(옵션)
    ap.add_argument("--estimate-z", action="store_true", help="KDTree로 z 추정 (켜면 라벨 cz 무시)")
    ap.add_argument("--z-radius", type=float, default=0.8, help="z 추정 반경(m)")
    ap.add_argument("--z-offset", type=float, default=-1, help="추정/라벨 z에 추가 오프셋(m)")

    # 크기 스케일링 모드
    ap.add_argument("--size-mode", choices=["dynamic", "fixed"], default="fixed",
                    help="dynamic: BEV 라벨의 length/width로 메쉬 스케일, fixed: 고정 값 사용")
    ap.add_argument("--fixed-length", type=float, default=5,
                    help="--size-mode fixed 일 때 사용할 고정 길이(m)")
    ap.add_argument("--fixed-width", type=float, default=4,
                    help="--size-mode fixed 일 때 사용할 고정 폭(m)")
    ap.add_argument("--height-scale", type=float, default=1,
                    help="차량 높이 스케일 = width * height_scale (기본 0.5). 시각화 높이와 바닥 오프셋 모두에 적용")

    # 성능/제어
    ap.add_argument("--max-cars", type=int, default=-1, help="사전 슬롯 상한(기본: 데이터의 최대값)")
    ap.add_argument("--unlit", action="store_true", help="차량 재질을 defaultUnlit로(조명계산 제거)")

    args = ap.parse_args()

    # ---- PLY 로드 ----
    print("[+] Load global cloud…")
    cloud = o3d.io.read_point_cloud(args.global_ply)
    if cloud.is_empty():
        raise RuntimeError(f"Empty point cloud: {args.global_ply}")
    bb = cloud.get_axis_aligned_bounding_box()
    print(f"    Cloud bounds XYZ: {np.asarray(bb.get_min_bound())} {np.asarray(bb.get_max_bound())}")

    # ---- KDTree (옵션) ----
    if args.estimate_z:
        print("[+] Build KDTree…")
        kdtree, xyz_pts = build_kdtree_for_z(cloud)
    else:
        kdtree, xyz_pts = (None, None)

    # ---- 차량 메쉬 로드 및 유닛화 ----
    print("[+] Load vehicle mesh…")
    mesh_ref = o3d.io.read_triangle_mesh(args.vehicle_glb, enable_post_processing=True)
    if mesh_ref.is_empty():
        raise RuntimeError(f"Cannot load mesh: {args.vehicle_glb}")
    mesh_unit = unitize_mesh(mesh_ref)

    # ---- 라벨 로드 ----
    print("[+] Read BEV labels…")
    frames = load_labels_dir(args.bev_label_dir)
    print(f"    Loaded {len(frames)} label frames")

    # 사전 슬롯 개수 산정
    if args.max_cars > 0:
        max_cars = args.max_cars
    else:
        max_cars = max((arr.shape[0] for _, arr in frames), default=0)
    print(f"[i] Pre-alloc vehicle slots: {max_cars}")

    # ============ O3DVisualizer 경로 ============
    use_modern = (not args.force_legacy) and hasattr(o3d.visualization, "O3DVisualizer")
    if use_modern:
        # GUI 초기화
        try:
            gui = o3d.visualization.gui
            app = gui.Application.instance
            try:
                app.initialize()
            except Exception:
                pass
        except Exception as e:
            print(f"[warn] GUI initialize 실패 → 레거시 폴백: {e}")
            use_modern = False

    if use_modern:
        vis = o3d.visualization.O3DVisualizer("PLY + GLB Overlay (optimized)", 1600, 1000)
        app.add_window(vis)

        # 안전한 종료 플래그
        _should_close = {"flag": False}
        def _on_close():
            _should_close["flag"] = True
            # 타이머/스레드 정리 (아래에서 상태 키가 있을 때만)
            try:
                t = state.get("timer", None)
                if t is not None and hasattr(t, "stop"):
                    t.stop()
            except Exception:
                pass
            state["running"] = False
            try:
                app.quit()
            except Exception:
                pass
            return True
        try:
            vis.set_on_close(_on_close)
        except Exception:
            pass

        # 배경/세팅
        try:
            vis.show_settings = False
        except Exception:
            pass
        try:
            if hasattr(vis, "show_skybox"):
                vis.show_skybox(False)
        except Exception:
            pass
        if args.bg_dark:
            for meth in ("set_background", "set_background_color"):
                if hasattr(vis, meth):
                    try:
                        getattr(vis, meth)((0, 0, 0, 1))
                        break
                    except Exception:
                        pass

        # 글로벌 클라우드 1회 추가
        vis.add_geometry("global_cloud", cloud)
        if args.invert_ply_y:
            try:
                vis.scene.set_geometry_transform("global_cloud", _flip_y_T())
            except Exception:
                pass

        # 차량 슬롯 사전 생성
        car_names = []
        material = None
        if args.unlit:
            try:
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultUnlit"
            except Exception:
                material = None

        for i in range(max_cars):
            name = f"car_{i:04d}"
            try:
                mesh_copy = o3d.geometry.TriangleMesh(mesh_unit)
                if material:
                    vis.add_geometry(name, mesh_copy, material)
                else:
                    vis.add_geometry(name, mesh_copy)
                if hasattr(vis, "scene"):
                    try:
                        vis.scene.set_geometry_is_visible(name, False)
                    except Exception:
                        pass
            except Exception:
                continue
            car_names.append(name)

        # 카메라 프레임 상태
        state = {
            "idx": 0,
            "paused": (not args.play),
            "last_tick": time.time(),
            "period": 1.0 / max(1e-6, args.fps),
        }

        FAR_HIDE_T = np.diag([1e-6, 1e-6, 1e-6, 1.0]).astype(np.float64)
        FAR_HIDE_T[:3, 3] = np.array([0.0, 0.0, -9999.0], dtype=np.float64)

        def _set_visible(name: str, visible: bool):
            # 1) 시도: 진짜 visibility 토글
            if hasattr(vis, "scene"):
                try:
                    vis.scene.set_geometry_is_visible(name, bool(visible))
                    return
                except Exception:
                    pass
            # 2) 폴백: 숨김 시 멀리/초소형으로 이동
            if not visible:
                try:
                    if hasattr(vis, "scene"):
                        vis.scene.set_geometry_transform(name, FAR_HIDE_T)
                    else:
                        # 레거시에선 여기 안옴
                        pass
                except Exception:
                    pass

        def apply_frame(idx: int):
            # 먼저 모든 슬롯을 숨김(유령 잔상 방지)
            for nm in car_names:
                _set_visible(nm, False)

            if idx < 0 or idx >= len(frames):
                return

            _, arr = frames[idx]
            N = arr.shape[0] if arr is not None else 0

            # Z 추정 or 라벨 cz
            if N > 0 and args.estimate_z and (kdtree is not None):
                zs_est = estimate_z_from_cloud(arr[:, 1:3], kdtree, xyz_pts,
                                               radius=args.z_radius, default_z=0.0) + float(args.z_offset)
            else:
                zs_est = None  # 사용 안함

            # 가시성/변환 갱신
            for i in range(min(N, len(car_names))):
                row = arr[i]
                if row.shape[0] >= 9:
                    # cls, cx, cy, cz, L, W, yaw, pitch, roll
                    _, cx, cy, cz_lab, length, width, yaw_deg, pitch_deg, roll_deg = row[:9]
                else:
                    # 이 경우는 거의 없음(로더에서 보정) — 안전망
                    _, cx, cy, length, width, yaw_deg = row[:6]
                    cz_lab, pitch_deg, roll_deg = (0.0, 0.0, 0.0)

                # 옵션: BEV Y축 반전 (라벨 좌표계를 월드에 맞추기)
                if args.invert_bev_y:
                    cy = -float(cy)
                    yaw_deg = -float(yaw_deg)
                    pitch_deg = -float(pitch_deg)   # ← 추가
                    roll_deg  = -float(roll_deg)    # ← 추가

                z_here = (zs_est[i] if (zs_est is not None and i < len(zs_est)) else float(cz_lab) + float(args.z_offset))

                # 스케일링 모드 적용: dynamic(라벨 기반) vs fixed(고정값)
                if args.size_mode == "fixed":
                    length_use = float(args.fixed_length)
                    width_use  = float(args.fixed_width)
                else:
                    length_use = float(length)
                    width_use  = float(width)

                height = width_use * float(args.height_scale / 2)
                center = np.array([cx, cy, z_here + height / 2], dtype=np.float32)
                T = build_unit_to_world_T(length_use, width_use, yaw_deg, center,
                                          pitch_deg=float(pitch_deg), roll_deg=float(roll_deg),
                                          up_scale_from_width=float(args.height_scale))
                _set_visible(car_names[i], True)
                try:
                    vis.scene.set_geometry_transform(car_names[i], T)
                except Exception:
                    pass

            for i in range(N, len(car_names)):
                _set_visible(car_names[i], False)

            try:
                vis.post_redraw()
            except Exception:
                pass

        # 초기 프레임 적용
        apply_frame(state["idx"])
        try:
            vis.reset_camera_to_default()
            vis.fit_geometry_to_view("global_cloud")
        except Exception:
            pass

        # 키 콜백
        def cb_left(_):
            state["idx"] = max(0, state["idx"] - 1)
            state["paused"] = True
            apply_frame(state["idx"])
            return True

        def cb_right(_):
            if len(frames) > 0:
                state["idx"] = min(len(frames) - 1, state["idx"] + 1)
            state["paused"] = True
            apply_frame(state["idx"])
            return True

        def cb_space(_):
            state["paused"] = not state["paused"]
            return True

        try:
            vis.add_key_event_callback(ord("A"), cb_left)
            vis.add_key_event_callback(ord("D"), cb_right)
            vis.add_key_event_callback(ord(" "), cb_space)
        except Exception:
            print("[warn] add_key_event_callback 미지원 버전입니다. 키 제어 없이 재생만 동작할 수 있어요.")

        # 타이머 기반 메인 루프 (GUI 스레드 주기 콜백)
        def _on_timer():
            now = time.time()
            if (not state["paused"]) and (now - state["last_tick"] >= state["period"]):
                state["last_tick"] = now
                if len(frames) > 0:
                    state["idx"] = (state["idx"] + 1) % len(frames)
                    if args.pause_on_empty and frames[state["idx"]][1].shape[0] == 0:
                        state["paused"] = True
                    apply_frame(state["idx"])

        # 메인 루프 구동: Open3D 버전별 호환 처리
        try:
            # 신버전: gui.Timer 존재
            state["timer"] = gui.Timer(state["period"], _on_timer)
            state["timer"].start()
        except AttributeError:
            # 구버전: gui.Timer가 없음 → 백그라운드 스레드로 주기 콜백을 메인스레드에 포스트
            state["running"] = True
            def _pump_loop():
                # 약간 더 촘촘히 깨워서 프레임 드롭 완화
                sleep_dt = max(0.005, state["period"] * 0.5)
                while state.get("running", False) and not _should_close["flag"]:
                    time.sleep(sleep_dt)
                    try:
                        app.post_to_main_thread(vis, _on_timer)
                    except Exception:
                        break
            state["pump_thread"] = threading.Thread(target=_pump_loop, daemon=True)
            state["pump_thread"].start()

        # GUI 이벤트 루프 실행 (윈도우가 닫히면 run()이 종료됨)
        app.run()

        # 종료 후 정리
        state["running"] = False
        try:
            t = state.get("timer", None)
            if t is not None and hasattr(t, "stop"):
                t.stop()
        except Exception:
            pass
        thr = state.get("pump_thread", None)
        if isinstance(thr, threading.Thread) and thr.is_alive():
            try:
                thr.join(timeout=0.5)
            except Exception:
                pass
        return

    # ============ 레거시 Visualizer 폴백 ============
    print("[info] O3DVisualizer 미지원/초기화 실패 → 레거시 Visualizer 폴백")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PLY + GLB Overlay (legacy)", width=1400, height=900)
    cloud_to_add = cloud
    if args.invert_ply_y:
        cloud_to_add = o3d.geometry.PointCloud(cloud)
        cloud_to_add.transform(_flip_y_T())
    vis.add_geometry(cloud_to_add)

    try:
        opt = vis.get_render_option()
        if args.bg_dark:
            opt.background_color = np.array([0, 0, 0])
    except Exception:
        pass

    FAR_HIDE_T = np.diag([1e-6, 1e-6, 1e-6, 1.0]).astype(np.float64)
    FAR_HIDE_T[:3, 3] = np.array([0.0, 0.0, -9999.0], dtype=np.float64)

    # ---- 차량 슬롯(레거시) 사전 생성 & in-place 업데이트 방식 ----
    base_mesh = o3d.geometry.TriangleMesh(mesh_unit)  # 단일 기준 메쉬(유닛)
    base_vertices = np.asarray(base_mesh.vertices).copy()
    base_triangles = np.asarray(base_mesh.triangles).copy()
    base_has_normals = base_mesh.has_vertex_normals()
    if not base_has_normals:
        base_mesh.compute_vertex_normals()
    base_normals = np.asarray(base_mesh.vertex_normals).copy()

    # 레거시에서는 set_geometry_transform 가 없으므로, 슬롯 메쉬들을 미리 add 한 뒤
    # 매 프레임마다 같은 객체의 vertices/triangles 를 원본으로 되돌리고 transform()을 적용,
    # vis.update_geometry(...) 만 호출한다.
    max_slots = max((arr.shape[0] for _, arr in frames), default=0)
    car_meshes = []
    for i in range(max_slots):
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(base_vertices.copy())
        m.triangles = o3d.utility.Vector3iVector(base_triangles.copy())
        if base_has_normals:
            m.vertex_normals = o3d.utility.Vector3dVector(base_normals.copy())
        # 처음에는 멀리 보내 숨김
        m.transform(FAR_HIDE_T)
        vis.add_geometry(m, reset_bounding_box=False)
        car_meshes.append(m)

    idx = 0
    paused = (not args.play)
    period = 1.0 / max(1e-6, args.fps)
    last_tick = time.time()
    running = True

    def _apply_to_slot(slot_mesh: o3d.geometry.TriangleMesh, T: np.ndarray):
        # 원본으로 되돌린 후 변환 적용
        slot_mesh.vertices = o3d.utility.Vector3dVector(base_vertices.copy())
        slot_mesh.triangles = o3d.utility.Vector3iVector(base_triangles.copy())
        if base_has_normals:
            slot_mesh.vertex_normals = o3d.utility.Vector3dVector(base_normals.copy())
        slot_mesh.transform(T)
        vis.update_geometry(slot_mesh)

    def set_frame_legacy(i):
        # 범위 체크
        if i < 0 or i >= len(frames):
            # 모두 숨김
            for m in car_meshes:
                _apply_to_slot(m, FAR_HIDE_T)
            return

        _, arr = frames[i]
        N = 0 if arr is None else int(arr.shape[0])

        # Z 추정 or 라벨 cz 사용
        if N > 0 and args.estimate_z and (kdtree is not None):
            zs_est = estimate_z_from_cloud(arr[:, 1:3], kdtree, xyz_pts,
                                           radius=args.z_radius, default_z=0.0) + float(args.z_offset)
        else:
            zs_est = None

        # 각 슬롯 갱신
        for k in range(len(car_meshes)):
            if k < N:
                row = arr[k]
                if row.shape[0] >= 9:
                    # cls, cx, cy, cz, L, W, yaw, pitch, roll
                    _, cx, cy, cz_lab, length, width, yaw_deg, pitch_deg, roll_deg = row[:9]
                else:
                    _, cx, cy, length, width, yaw_deg = row[:6]
                    cz_lab, pitch_deg, roll_deg = (0.0, 0.0, 0.0)

                # 옵션: BEV Y축 반전(월드 좌표 일치) — yaw/pitch/roll 모두 부호 반전
                if args.invert_bev_y:
                    cy        = -float(cy)
                    yaw_deg   = -float(yaw_deg)
                    pitch_deg = -float(pitch_deg)
                    roll_deg  = -float(roll_deg)

                z_here = (zs_est[k] if (zs_est is not None and k < len(zs_est)) else float(cz_lab) + float(args.z_offset))

                # 크기 스케일링
                if args.size_mode == "fixed":
                    length_use = float(args.fixed_length)
                    width_use  = float(args.fixed_width)
                else:
                    length_use = float(length)
                    width_use  = float(width)

                height = width_use * float(args.height_scale)
                center = np.array([cx, cy, z_here + height / 2], dtype=np.float32)

                T = build_unit_to_world_T(length_use, width_use, yaw_deg, center,
                                          pitch_deg=float(pitch_deg), roll_deg=float(roll_deg),
                                          up_scale_from_width=float(args.height_scale))
                _apply_to_slot(car_meshes[k], T)
            else:
                # 남은 슬롯은 숨김 위치로 이동
                _apply_to_slot(car_meshes[k], FAR_HIDE_T)

    # ---- 키 입력 콜백 (레거시 전용) ----
    def _cb_space(vis_):
        nonlocal paused
        paused = not paused
        return False  # False: 다른 핸들러로 이벤트 전달 허용

    def _cb_left(vis_):
        nonlocal idx, paused, last_tick
        if len(frames) > 0:
            idx = max(0, idx - 1)
            paused = True
            set_frame_legacy(idx)
            last_tick = time.time()
        return False

    def _cb_right(vis_):
        nonlocal idx, paused, last_tick
        if len(frames) > 0:
            idx = min(len(frames) - 1, idx + 1)
            paused = True
            set_frame_legacy(idx)
            last_tick = time.time()
        return False

    def _cb_quit(vis_):
        nonlocal running
        running = False
        return False

    # 스페이스(재생/일시정지), A/D(이전/다음), Q(종료)
    vis.register_key_callback(ord(" "), _cb_space)
    vis.register_key_callback(ord("A"), _cb_left)
    vis.register_key_callback(ord("D"), _cb_right)
    vis.register_key_callback(ord("Q"), _cb_quit)

    set_frame_legacy(idx)
    while running:
        now = time.time()
        if (not paused) and (now - last_tick >= period):
            last_tick = now
            if len(frames) > 0:
                idx = (idx + 1) % len(frames)
                if args.pause_on_empty and frames[idx][1].shape[0] == 0:
                    paused = True
                set_frame_legacy(idx)

        # CPU 점유율 과다 방지 및 GUI 여유 확보
        time.sleep(max(0.0, period * 0.2))
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    main()


"""
python ./pointcloud/overlay_obj_on_ply.py \
  --global-ply ./pointcloud/cloud_rgb_ply/cloud_rgb_9.ply \
  --bev-label-dir ./dataset_exmple_pointcloud_9/bev_labels \
  --vehicle-glb ./pointcloud/car.glb \
  --fps 30 --play --bg-dark \
  --force-legacy  
"""