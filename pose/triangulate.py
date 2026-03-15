"""
多視点三角測量モジュール (DLT法)

各カメラの2Dキーポイントとカメラパラメータから
3D座標を復元する。
"""

from __future__ import annotations

import numpy as np


def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    射影行列 P = K @ [R | t] を構成する。

    Args:
        K: 内部パラメータ行列 (3, 3)
        R: 回転行列 (3, 3)
        t: 並進ベクトル (3, 1)

    Returns:
        射影行列 P (3, 4)
    """
    Rt = np.hstack([R, t.reshape(3, 1)])  # (3, 4)
    return K @ Rt


def triangulate_point(
    proj_matrices: list[np.ndarray],
    points_2d: list[np.ndarray],
    valid_flags: list[bool],
) -> np.ndarray | None:
    """
    DLT法で1点の3D座標を復元する。

    Args:
        proj_matrices: 各カメラの射影行列リスト [(3,4), ...]
        points_2d: 各カメラでの対応2D点 [(2,), ...]
        valid_flags: 各カメラで有効かどうかのフラグ

    Returns:
        3D座標 (3,) または None（有効なカメラが2台未満の場合）
    """
    valid_projs = [P for P, v in zip(proj_matrices, valid_flags) if v]
    valid_pts = [pt for pt, v in zip(points_2d, valid_flags) if v]

    if len(valid_projs) < 2:
        return None

    # DLT: Ax = 0 の形に変換
    A_rows = []
    for P, pt in zip(valid_projs, valid_pts):
        x, y = float(pt[0]), float(pt[1])
        A_rows.append(x * P[2] - P[0])
        A_rows.append(y * P[2] - P[1])

    A = np.stack(A_rows, axis=0)  # (2n, 4)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # 最小特異値に対応する行ベクトル
    X = X / X[3]  # 同次座標を正規化

    return X[:3].astype(np.float32)


def triangulate_pose(
    proj_matrices: list[np.ndarray],
    poses_per_camera: list[np.ndarray | None],
    score_threshold: float = 0.3,
    scores_per_camera: list[np.ndarray | None] | None = None,
) -> np.ndarray:
    """
    複数カメラの2Dキーポイント列から133点の3D座標を一括復元する。

    バッチSVD最適化: 全カメラで有効なキーポイントをまとめてnp.linalg.svdに渡し、
    Pythonループ133回を廃止する。

    Args:
        proj_matrices: 各カメラの射影行列 [(3,4), ...] (カメラ数 N)
        poses_per_camera: 各カメラの2Dキーポイント配列 [(133,2) or None, ...]
        score_threshold: この値未満のキーポイントは無効とみなす
        scores_per_camera: 各カメラのスコア配列 [(133,) or None, ...]

    Returns:
        3Dキーポイント配列 (133, 3)。無効な点は NaN。
    """
    n_kps = 133
    n_cams = len(proj_matrices)
    out = np.full((n_kps, 3), np.nan, dtype=np.float32)

    if scores_per_camera is None:
        scores_per_camera = [None] * n_cams

    # ── キーポイントごとの有効フラグを一括計算 ──────────────────────────────
    # valid[c, k] = True ⟺ カメラc でキーポイントk が有効
    valid = np.zeros((n_cams, n_kps), dtype=bool)
    kps_list: list[np.ndarray | None] = []

    for c, (pose, scores) in enumerate(zip(poses_per_camera, scores_per_camera)):
        if pose is None:
            kps_list.append(None)
            continue
        kps_list.append(pose)
        v = (pose[:, 0] >= 0) & (pose[:, 1] >= 0)  # (n_kps,)
        if scores is not None:
            v = v & (scores >= score_threshold)
        valid[c] = v

    # ── 全カメラ有効なキーポイントをバッチSVDで一括処理 ──────────────────────
    all_valid_mask = valid.all(axis=0)  # (n_kps,) - 全カメラで有効
    batch_indices = np.where(all_valid_mask)[0]

    if len(batch_indices) > 0:
        # A_batch shape: (n_batch, 2*n_cams, 4)
        n_batch = len(batch_indices)
        A_batch = np.zeros((n_batch, 2 * n_cams, 4), dtype=np.float64)

        for c, (P, pose) in enumerate(zip(proj_matrices, kps_list)):
            if pose is None:
                continue
            pts = pose[batch_indices]  # (n_batch, 2)
            x = pts[:, 0:1]  # (n_batch, 1)
            y = pts[:, 1:2]
            A_batch[:, 2 * c] = x * P[2] - P[0]  # (n_batch, 4)
            A_batch[:, 2 * c + 1] = y * P[2] - P[1]

        # バッチSVD: Vt shape (n_batch, 4, 4) ※ 2*n_cams >= 4 の場合
        _, _, Vt = np.linalg.svd(A_batch)  # Vt: (n_batch, min_dim, 4)
        X = Vt[:, -1, :]  # (n_batch, 4) 最小特異値ベクトル
        w = X[:, 3:4]
        nonzero = np.abs(w) > 1e-10
        X_norm = np.where(nonzero, X / np.where(nonzero, w, 1.0), np.nan)
        out[batch_indices] = X_norm[:, :3].astype(np.float32)

    # ── 一部カメラのみ有効なキーポイントを個別処理（フォールバック） ──────────
    partial_mask = (~all_valid_mask) & (valid.sum(axis=0) >= 2)
    for kp_idx in np.where(partial_mask)[0]:
        pts_2d: list[np.ndarray] = []
        flags: list[bool] = []
        for c, (pose, scores) in enumerate(zip(kps_list, scores_per_camera)):
            if pose is None:
                pts_2d.append(np.zeros(2, dtype=np.float32))
                flags.append(False)
                continue
            pt = pose[kp_idx]
            is_valid = bool(valid[c, kp_idx])
            pts_2d.append(pt)
            flags.append(is_valid)
        pt3d = triangulate_point(proj_matrices, pts_2d, flags)
        if pt3d is not None:
            out[kp_idx] = pt3d

    return out


def build_proj_matrices_from_params(params: dict) -> dict[int, np.ndarray]:
    """
    camera_params.json のパラメータから射影行列を一括構成する。

    Args:
        params: load_params() で読み込んだ辞書

    Returns:
        camera_index -> 射影行列 (3, 4) のマッピング
    """
    result: dict[int, np.ndarray] = {}
    intrinsics = params["intrinsics"]
    extrinsics = params["extrinsics"]

    for cam_str in intrinsics:
        cam_idx = int(cam_str)
        K = np.array(intrinsics[cam_str]["K"], dtype=np.float64)
        R = np.array(extrinsics[cam_str]["R"], dtype=np.float64)
        t = np.array(extrinsics[cam_str]["t"], dtype=np.float64).reshape(3, 1)
        result[cam_idx] = build_projection_matrix(K, R, t)

    return result
