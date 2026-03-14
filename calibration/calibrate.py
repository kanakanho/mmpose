"""
カメラキャリブレーションモジュール

チェッカーボードを使って各カメラの内部パラメータと外部パラメータを計算し、
camera_params.json に保存する。

使用方法:
    python main.py --mode calibrate --cameras 0 1 --board-size 9 6 --square-size 25.0
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PARAMS_PATH = Path(__file__).parent.parent / "camera_params.json"


def capture_calibration_frames(
    camera_indices: list[int],
    board_size: tuple[int, int] = (9, 6),
    n_frames: int = 20,
) -> dict[int, list[np.ndarray]]:
    """
    複数カメラから同時にチェッカーボード画像を収集する。

    Args:
        camera_indices: カメラインデックスのリスト
        board_size: チェッカーボードの内角数 (cols, rows)
        n_frames: 収集するフレーム数

    Returns:
        camera_index -> frames のマッピング
    """
    caps = {idx: cv2.VideoCapture(idx) for idx in camera_indices}
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frames: dict[int, list[np.ndarray]] = {idx: [] for idx in camera_indices}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(
        f"[Calibration] チェッカーボード({board_size[0]}x{board_size[1]})を各カメラに映してください。"
    )
    print(f"[Calibration] 'c' で撮影、'q' で終了。目標: {n_frames} フレーム")

    while True:
        # 全カメラからフレームを取得
        raw: dict[int, np.ndarray] = {}
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                raw[idx] = frame

        if len(raw) < len(camera_indices):
            continue

        # プレビュー表示（全カメラを横並び）
        grays = {idx: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for idx, f in raw.items()}
        previews = []
        for idx in camera_indices:
            gray = grays[idx]
            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            preview = raw[idx].copy()
            if found:
                cv2.drawChessboardCorners(preview, board_size, corners, found)
            count_txt = f"Cam{idx}: {len(frames[idx])}/{n_frames}"
            cv2.putText(
                preview,
                count_txt,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            previews.append(cv2.resize(preview, (640, 360)))

        combined = np.hstack(previews)
        cv2.imshow("Calibration Capture", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[Calibration] 中断しました。")
            break
        elif key == ord("c"):
            # 全カメラでチェッカーボードを検出できた場合のみ保存
            all_found = True
            found_corners: dict[int, np.ndarray] = {}
            for idx in camera_indices:
                gray = grays[idx]
                found, corners = cv2.findChessboardCorners(gray, board_size, None)
                if found:
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    found_corners[idx] = corners2
                else:
                    all_found = False
                    print(
                        f"[Calibration] カメラ{idx}でチェッカーボードが見つかりません。"
                    )
                    break

            if all_found:
                for idx in camera_indices:
                    frames[idx].append((raw[idx], found_corners[idx]))
                count = len(frames[camera_indices[0]])
                print(f"[Calibration] {count}/{n_frames} フレーム収集済み")
                if count >= n_frames:
                    print("[Calibration] 必要なフレーム数に達しました。")
                    break

    cv2.destroyAllWindows()
    for cap in caps.values():
        cap.release()

    return frames


def calibrate_cameras(
    frames: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> dict:
    """
    収集したフレームからカメラパラメータを計算する。

    Args:
        frames: camera_index -> [(image, corners), ...] のマッピング
        board_size: チェッカーボードの内角数 (cols, rows)
        square_size_mm: チェッカーボードの1マスのサイズ [mm]

    Returns:
        カメラパラメータ辞書
    """
    camera_indices = sorted(frames.keys())

    # 3Dオブジェクト点（チェッカーボード上の座標）を準備
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = [objp] * len(frames[camera_indices[0]])

    # 各カメラの内部パラメータをキャリブレーション
    intrinsics: dict[int, dict] = {}
    for idx in camera_indices:
        img_points = [corners for _, corners in frames[idx]]
        h, w = frames[idx][0][0].shape[:2]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        intrinsics[idx] = {
            "K": K.tolist(),
            "dist": dist.tolist(),
            "rms": float(rms),
            "image_size": [w, h],
        }
        print(f"[Calibration] カメラ{idx} 内部パラメータ RMS={rms:.3f}px")

    # カメラ間の外部パラメータ（カメラ0を基準座標系）
    extrinsics: dict[int, dict] = {}
    # カメラ0 の外部パラメータは単位行列
    extrinsics[camera_indices[0]] = {
        "R": np.eye(3).tolist(),
        "t": np.zeros((3, 1)).tolist(),
    }

    ref_idx = camera_indices[0]
    ref_img_points = [corners for _, corners in frames[ref_idx]]
    K0 = np.array(intrinsics[ref_idx]["K"])
    dist0 = np.array(intrinsics[ref_idx]["dist"])

    for idx in camera_indices[1:]:
        img_points_i = [corners for _, corners in frames[idx]]
        Ki = np.array(intrinsics[idx]["K"])
        disti = np.array(intrinsics[idx]["dist"])

        rms, K0_new, dist0_new, Ki_new, disti_new, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            ref_img_points,
            img_points_i,
            K0,
            dist0,
            Ki,
            disti,
            frames[ref_idx][0][0].shape[:2][::-1],
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        extrinsics[idx] = {
            "R": R.tolist(),
            "t": T.tolist(),
        }
        print(
            f"[Calibration] カメラ{ref_idx}→{idx} ステレオキャリブレーション RMS={rms:.3f}px"
        )

    params = {
        "camera_indices": camera_indices,
        "board_size": list(board_size),
        "square_size_mm": square_size_mm,
        "intrinsics": {str(k): v for k, v in intrinsics.items()},
        "extrinsics": {str(k): v for k, v in extrinsics.items()},
    }
    return params


def save_params(params: dict, path: Path = DEFAULT_PARAMS_PATH) -> None:
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False))
    print(f"[Calibration] パラメータを保存しました: {path}")


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"カメラパラメータファイルが見つかりません: {path}\n先に --mode calibrate を実行してください。"
        )
    return json.loads(path.read_text())


def run_calibration(
    camera_indices: list[int],
    board_size: tuple[int, int] = (9, 6),
    n_frames: int = 20,
    square_size_mm: float = 25.0,
    out_path: Path = DEFAULT_PARAMS_PATH,
) -> dict:
    """キャリブレーションのフルパイプラインを実行する。"""
    frames = capture_calibration_frames(camera_indices, board_size, n_frames)
    if any(len(v) == 0 for v in frames.values()):
        raise RuntimeError(
            "フレームが収集できませんでした。キャリブレーションを中断します。"
        )
    params = calibrate_cameras(frames, board_size, square_size_mm)
    save_params(params, out_path)
    return params
