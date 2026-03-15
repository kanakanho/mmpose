"""
MMPose マルチカメラ全身3D検出 + OSC配信 エントリーポイント

使用方法:
  # キャリブレーション (カメラ0,1 使用)
  python main.py --mode calibrate --cameras 0 1

  # 推論実行
  python main.py --mode run --cameras 0 1 --osc-host 127.0.0.1 --osc-port 9000

  # キャリブレーションなしで単体カメラ動作確認 (三角測量なし・2D表示のみ)
  python main.py --mode run --cameras 0 --no-calibration
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from calibration.calibrate import DEFAULT_PARAMS_PATH, load_params, run_calibration
from output.osc_sender import OSCSender
from output.preview import MultiCameraPreview
from pose.detector import PoseResult, WholebodyDetector
from pose.hand_splitter import split_keypoints
from pose.triangulate import build_proj_matrices_from_params, triangulate_pose


# ─── バックグラウンドカメラキャプチャスレッド ──────────────────────────────
class CameraThread:
    """
    cv2.VideoCapture をバックグラウンドスレッドで常時読み出し、
    最新フレームを保持する。これによりメインループのI/Oブロッキングをなくす。
    """

    def __init__(self, cap: cv2.VideoCapture, cam_idx: int) -> None:
        self.cam_idx = cam_idx
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stopped = True
        self._cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MMPose マルチカメラ全身3D検出 + OSC配信"
    )
    parser.add_argument(
        "--mode",
        choices=["calibrate", "run"],
        default="run",
        help="実行モード: calibrate=キャリブレーション, run=推論",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        type=int,
        default=[0],
        metavar="CAM_IDX",
        help="使用するカメラのインデックス (例: 0 1 2)",
    )
    # ─── キャリブレーション設定 ───────────────────────────────────────────────
    parser.add_argument(
        "--board-size",
        nargs=2,
        type=int,
        default=[9, 6],
        metavar=("COLS", "ROWS"),
        help="チェッカーボードの内角数 (default: 9 6)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="チェッカーボードの1マスのサイズ [mm] (default: 25.0)",
    )
    parser.add_argument(
        "--calib-frames",
        type=int,
        default=20,
        help="キャリブレーションに使うフレーム数 (default: 20)",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="カメラパラメータの保存/読み込みパス",
    )
    # ─── 推論設定 ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="キャリブレーションなし (単体カメラ・2D表示のみ)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="キーポイントの信頼度閾値 (default: 0.3)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="推論デバイス (default: cpu / cudaが使える場合は cuda:0 を指定)",
    )
    # ─── OSC 設定 ────────────────────────────────────────────────────────────
    parser.add_argument("--osc-host", default="127.0.0.1", help="OSC送信先ホスト")
    parser.add_argument("--osc-port", type=int, default=9000, help="OSC送信先ポート")
    parser.add_argument(
        "--osc-scale",
        type=float,
        default=0.001,
        help="座標スケール係数 (default: 0.001 = mm→m)",
    )
    # ─── プレビュー設定 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="OpenCVプレビューを無効化",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=640,
        help="プレビュー1カメラあたりの表示幅 (default: 640)",
    )
    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> None:
    """推論メインループ"""
    camera_indices = args.cameras

    # ─── カメラ起動 (バックグラウンドスレッド) ──────────────────────────────
    cam_threads: dict[int, CameraThread] = {}
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファを最小化して古フレームを防ぐ
        if not cap.isOpened():
            raise RuntimeError(f"カメラ {idx} が開けません。")
        cam_threads[idx] = CameraThread(cap, idx)
    print(
        f"[Main] カメラ {camera_indices} を起動しました（バックグラウンドスレッド）。"
    )

    # ─── カメラパラメータ読み込み ────────────────────────────────────────────
    use_3d = not args.no_calibration and len(camera_indices) >= 2
    proj_matrices: dict[int, np.ndarray] = {}
    if use_3d:
        params = load_params(args.params_path)
        proj_matrices = build_proj_matrices_from_params(params)
        print("[Main] カメラパラメータを読み込みました。3D三角測量を実行します。")
    else:
        if len(camera_indices) < 2:
            print("[Main] カメラが1台のため 2D キーポイント出力モードで動作します。")
        else:
            print(
                "[Main] --no-calibration が指定されました。2D出力モードで動作します。"
            )

    # ─── モジュール初期化 ────────────────────────────────────────────────────
    # カメラ台数分のDetectorを生成（並列推論のためスレッドセーフに各インスタンスを使用）
    detectors: dict[int, WholebodyDetector] = {
        idx: WholebodyDetector(device=args.device, score_threshold=args.score_threshold)
        for idx in camera_indices
    }
    # MMPoseのレジストリ競合を防ぐため、並列推論開始前に順番にロードする
    print("[Main] MMPose モデルを事前ロード中...")
    for idx in camera_indices:
        detectors[idx]._load()
    print("[Main] 全モデルのロード完了")

    osc = OSCSender(
        host=args.osc_host, port=args.osc_port, coordinate_scale=args.osc_scale
    )
    preview: MultiCameraPreview | None = None
    if not args.no_preview:
        preview = MultiCameraPreview(display_width=args.preview_width)

    # ─── 推論ループ ──────────────────────────────────────────────────────────
    frame_idx = 0
    prev_time = time.perf_counter()

    print("[Main] 推論開始。ESCキーで終了。")
    try:
        # 推論用リサイズ解像度（人物検出コスト削減）
        INFER_W, INFER_H = 640, 360

        # スレッドプールをループ外で1度だけ生成（毎フレームの生成コストを排除）
        with ThreadPoolExecutor(max_workers=len(camera_indices)) as executor:
            while True:
                # 全カメラから最新フレームを取得（バックグラウンドスレッドからノンブロッキング）
                frames: dict[int, np.ndarray] = {}
                for idx, ct in cam_threads.items():
                    f = ct.get_frame()
                    if f is not None:
                        frames[idx] = f

                if len(frames) < len(camera_indices):
                    time.sleep(0.001)
                    continue

                # 推論用に縮小フレームを作成（キーポイント座標は縮小後の空間）
                small_frames: dict[int, np.ndarray] = {
                    idx: cv2.resize(frame, (INFER_W, INFER_H))
                    for idx, frame in frames.items()
                }

                # 各カメラで2D骨格検出（スレッドプールで並列実行）
                results_per_cam: dict[int, PoseResult | None] = {}
                all_results_for_preview: list[list[PoseResult]] = []

                def _infer(idx: int) -> tuple[int, list[PoseResult]]:
                    return idx, detectors[idx].infer(small_frames[idx])

                fut_results = list(executor.map(_infer, camera_indices))

                for idx, det_results in fut_results:
                    if det_results:
                        results_per_cam[idx] = det_results[0]  # 1人分のみ
                        all_results_for_preview.append(det_results[:1])
                    else:
                        results_per_cam[idx] = None
                        all_results_for_preview.append([])

                # OSCフレーム開始通知
                osc.send_frame_start(frame_idx)

                if use_3d:
                    # 多視点三角測量で3D座標を復元
                    kps_per_cam: list[np.ndarray | None] = []
                    scores_per_cam: list[np.ndarray | None] = []
                    proj_list: list[np.ndarray] = []

                    for idx in camera_indices:
                        res = results_per_cam.get(idx)
                        if res is not None and idx in proj_matrices:
                            kps_per_cam.append(res.keypoints)
                            scores_per_cam.append(res.scores)
                            proj_list.append(proj_matrices[idx])
                        else:
                            kps_per_cam.append(None)
                            scores_per_cam.append(None)
                            # ダミーの射影行列（Noneの場合は三角測量で無視される）
                            proj_list.append(np.zeros((3, 4)))

                    kps3d = triangulate_pose(
                        proj_matrices=proj_list,
                        poses_per_camera=kps_per_cam,
                        score_threshold=args.score_threshold,
                        scores_per_camera=scores_per_cam,
                    )

                    # 部位ごとに分離してOSC送信
                    split = split_keypoints(kps3d)
                    osc.send_pose(split)

                else:
                    # 2Dモード: カメラ0の座標をそのままOSC送信 (z=0)
                    res0 = results_per_cam.get(camera_indices[0])
                    if res0 is not None:
                        kps2d = res0.keypoints  # (133, 2)
                        # 2D→3D: z=0 で拡張
                        kps3d_dummy = np.concatenate(
                            [kps2d, np.zeros((kps2d.shape[0], 1), dtype=np.float32)],
                            axis=1,
                        )
                        split = split_keypoints(kps3d_dummy, res0.scores)
                        osc.send_pose(split)

                # FPS計算
                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-9)
                prev_time = now

                # プレビュー表示（推論と同じ縮小フレームを使用：再描画コスト削減）
                if preview is not None:
                    frames_list = [
                        small_frames[idx]
                        for idx in camera_indices
                        if idx in small_frames
                    ]
                    should_continue = preview.show(
                        frames_list, all_results_for_preview, fps=fps
                    )
                    if not should_continue:
                        print("[Main] ESCキーで終了します。")
                        break

                frame_idx += 1

    finally:
        for ct in cam_threads.values():
            ct.stop()
        if preview is not None:
            preview.close()
        print("[Main] 終了しました。")


def main() -> None:
    args = parse_args()

    if args.mode == "calibrate":
        print(f"[Main] キャリブレーションモード開始 (カメラ: {args.cameras})")
        run_calibration(
            camera_indices=args.cameras,
            board_size=tuple(args.board_size),  # type: ignore[arg-type]
            n_frames=args.calib_frames,
            square_size_mm=args.square_size,
            out_path=args.params_path,
        )
    elif args.mode == "run":
        run_inference(args)


if __name__ == "__main__":
    main()
