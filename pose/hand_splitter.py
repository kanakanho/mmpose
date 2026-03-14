"""
COCO-WholeBody 133点キーポイントを体・顔・左手・右手に分離するモジュール

インデックス割り当て (COCO-WholeBody):
  0-16:   体 (17点) - COCO 17キーポイント
  17-22:  足先 (6点)
  23-90:  顔 (68点)
  91-111: 左手 (21点)
  112-132: 右手 (21点)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ─── インデックス範囲定義 ────────────────────────────────────────────────────
BODY_INDICES = list(range(0, 17))
FOOT_INDICES = list(range(17, 23))
FACE_INDICES = list(range(23, 91))
LEFT_HAND_INDICES = list(range(91, 112))
RIGHT_HAND_INDICES = list(range(112, 133))

# ─── 体のキーポイント名 (COCO 17点) ─────────────────────────────────────────
BODY_KEYPOINT_NAMES = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
]

# ─── 手のキーポイント名 (MediaPipe 21点, 左右共通) ──────────────────────────
HAND_KEYPOINT_NAMES = [
    "wrist",  # 0
    "thumb_cmc",  # 1
    "thumb_mcp",  # 2
    "thumb_ip",  # 3
    "thumb_tip",  # 4
    "index_mcp",  # 5
    "index_pip",  # 6
    "index_dip",  # 7
    "index_tip",  # 8
    "middle_mcp",  # 9
    "middle_pip",  # 10
    "middle_dip",  # 11
    "middle_tip",  # 12
    "ring_mcp",  # 13
    "ring_pip",  # 14
    "ring_dip",  # 15
    "ring_tip",  # 16
    "pinky_mcp",  # 17
    "pinky_pip",  # 18
    "pinky_dip",  # 19
    "pinky_tip",  # 20
]


@dataclass
class SplitPose:
    """部位ごとに分離された3D/2Dキーポイント"""

    body: np.ndarray  # (17, 3) or (17, 2) - 体
    foot: np.ndarray  # (6, 3) or (6, 2)   - 足先
    face: np.ndarray  # (68, 3) or (68, 2) - 顔ランドマーク
    left_hand: np.ndarray  # (21, 3) or (21, 2) - 左手
    right_hand: np.ndarray  # (21, 3) or (21, 2) - 右手

    # 信頼度スコア (2D検出時のみ)
    body_scores: np.ndarray | None = None  # (17,)
    face_scores: np.ndarray | None = None  # (68,)
    left_hand_scores: np.ndarray | None = None  # (21,)
    right_hand_scores: np.ndarray | None = None  # (21,)


def split_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray | None = None,
) -> SplitPose:
    """
    133点 (または 133×2/133×3) のキーポイント配列を部位ごとに分離する。

    Args:
        keypoints: shape (133, 2) または (133, 3)
        scores: shape (133,) の信頼度スコア (省略可)

    Returns:
        SplitPose
    """
    return SplitPose(
        body=keypoints[BODY_INDICES],
        foot=keypoints[FOOT_INDICES],
        face=keypoints[FACE_INDICES],
        left_hand=keypoints[LEFT_HAND_INDICES],
        right_hand=keypoints[RIGHT_HAND_INDICES],
        body_scores=scores[BODY_INDICES] if scores is not None else None,
        face_scores=scores[FACE_INDICES] if scores is not None else None,
        left_hand_scores=scores[LEFT_HAND_INDICES] if scores is not None else None,
        right_hand_scores=scores[RIGHT_HAND_INDICES] if scores is not None else None,
    )


def body_keypoints_dict(
    body_kps: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, dict]:
    """
    体キーポイントを名前付き辞書に変換する。

    Returns:
        {"nose": {"pos": [x,y,z], "score": 0.9}, ...}
    """
    result = {}
    for i, name in enumerate(BODY_KEYPOINT_NAMES):
        entry: dict = {"pos": body_kps[i].tolist()}
        if scores is not None:
            entry["score"] = float(scores[i])
        result[name] = entry
    return result


def hand_keypoints_dict(
    hand_kps: np.ndarray,
    side: str,
    scores: np.ndarray | None = None,
) -> dict[str, dict]:
    """
    手キーポイントを名前付き辞書に変換する。

    Args:
        side: "left" or "right"
    """
    result = {}
    for i, name in enumerate(HAND_KEYPOINT_NAMES):
        entry: dict = {"pos": hand_kps[i].tolist()}
        if scores is not None:
            entry["score"] = float(scores[i])
        result[name] = entry
    return result
