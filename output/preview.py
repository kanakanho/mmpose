"""
OpenCV プレビューモジュール

各カメラ映像に2Dキーポイントをオーバーレイし、横並び表示する。
体・顔・左手・右手を色分けして描画する。
"""

from __future__ import annotations

import cv2
import numpy as np

from pose.hand_splitter import (
    BODY_INDICES,
    FACE_INDICES,
    FOOT_INDICES,
    LEFT_HAND_INDICES,
    RIGHT_HAND_INDICES,
)
from pose.detector import PoseResult

# ─── 描画カラー (BGR) ────────────────────────────────────────────────────────
COLOR_BODY = (0, 255, 0)  # 緑
COLOR_FACE = (255, 200, 0)  # 青緑
COLOR_LEFT_HAND = (0, 100, 255)  # オレンジ
COLOR_RIGHT_HAND = (255, 50, 200)  # ピンク
COLOR_FOOT = (180, 180, 0)  # 黄

# ─── 体スケルトン接続 (COCO 17点) ───────────────────────────────────────────
BODY_SKELETON = [
    (0, 1),
    (0, 2),  # 鼻 → 目
    (1, 3),
    (2, 4),  # 目 → 耳
    (0, 5),
    (0, 6),  # 鼻 → 肩
    (5, 6),  # 肩-肩
    (5, 7),
    (7, 9),  # 左腕
    (6, 8),
    (8, 10),  # 右腕
    (5, 11),
    (6, 12),  # 胴体
    (11, 12),  # 腰-腰
    (11, 13),
    (13, 15),  # 左脚
    (12, 14),
    (14, 16),  # 右脚
]

# ─── 手スケルトン接続 (21点) ─────────────────────────────────────────────────
HAND_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # 親指
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # 人差し指
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # 中指
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # 薬指
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # 小指
]


def _draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 4,
) -> None:
    """キーポイントを点として描画"""
    for i in range(len(keypoints)):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        if x < 0 or y < 0:
            continue
        cv2.circle(frame, (x, y), radius, color, -1)


def _draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    connections: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """スケルトンをラインとして描画"""
    for i, j in connections:
        x1, y1 = int(keypoints[i, 0]), int(keypoints[i, 1])
        x2, y2 = int(keypoints[j, 0]), int(keypoints[j, 1])
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def draw_pose_on_frame(
    frame: np.ndarray,
    results: list[PoseResult],
    show_face: bool = True,
    show_hands: bool = True,
) -> np.ndarray:
    """
    検出結果をフレームにオーバーレイ描画する。

    Args:
        frame: BGR 画像 (H, W, 3)
        results: 検出された人物のリスト

    Returns:
        描画済み BGR 画像
    """
    vis = frame.copy()

    for result in results:
        kps = result.keypoints  # (133, 2)

        # 体
        body_kps = kps[BODY_INDICES]
        _draw_skeleton(vis, body_kps, BODY_SKELETON, COLOR_BODY, thickness=2)
        _draw_keypoints(vis, body_kps, COLOR_BODY)

        # 足先
        foot_kps = kps[FOOT_INDICES]
        _draw_keypoints(vis, foot_kps, COLOR_FOOT, radius=3)

        if show_face:
            # 顔
            face_kps = kps[FACE_INDICES]
            _draw_keypoints(vis, face_kps, COLOR_FACE, radius=2)

        if show_hands:
            # 左手
            lh_kps = kps[LEFT_HAND_INDICES]
            _draw_skeleton(vis, lh_kps, HAND_SKELETON, COLOR_LEFT_HAND, thickness=1)
            _draw_keypoints(vis, lh_kps, COLOR_LEFT_HAND, radius=3)

            # 右手
            rh_kps = kps[RIGHT_HAND_INDICES]
            _draw_skeleton(vis, rh_kps, HAND_SKELETON, COLOR_RIGHT_HAND, thickness=1)
            _draw_keypoints(vis, rh_kps, COLOR_RIGHT_HAND, radius=3)

    return vis


class MultiCameraPreview:
    """
    複数カメラ映像を横並びに表示する OpenCV ウィンドウ。

    Args:
        window_name: ウィンドウタイトル
        display_width: 各カメラ映像の表示幅（ピクセル）
        display_height: 各カメラ映像の表示高さ（ピクセル）
    """

    def __init__(
        self,
        window_name: str = "MMPose Multi-Camera",
        display_width: int = 640,
        display_height: int = 360,
    ):
        self.window_name = window_name
        self.display_width = display_width
        self.display_height = display_height
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show(
        self,
        frames: list[np.ndarray],
        all_results: list[list[PoseResult]],
        fps: float = 0.0,
    ) -> bool:
        """
        各カメラの映像と検出結果を表示する。

        Args:
            frames: 各カメラのBGR画像リスト
            all_results: 各カメラの検出結果リスト
            fps: FPS表示用の値

        Returns:
            False を返したら終了シグナル（ESCキー）
        """
        previews = []
        for frame, results in zip(frames, all_results):
            vis = draw_pose_on_frame(frame, results)
            if fps > 0:
                cv2.putText(
                    vis,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            previews.append(cv2.resize(vis, (self.display_width, self.display_height)))

        combined = np.hstack(previews)
        cv2.imshow(self.window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        return key != 27  # ESCで終了

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
