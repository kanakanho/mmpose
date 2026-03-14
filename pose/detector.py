"""
MMPose 推論モジュール

RTMPose wholebody (COCO-WholeBody 133点) により
体・顔・両手のキーポイントを各カメラフレームから検出する。

CPU 動作を想定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseResult:
    """1人分の推論結果"""

    keypoints: np.ndarray  # shape (133, 2) - 2D座標 [x, y]
    scores: np.ndarray  # shape (133,)   - 信頼度スコア
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))  # [x1, y1, x2, y2]


class WholebodyDetector:
    """
    MMPoseInferencer の wholebody モデルをラップし、
    各フレームから PoseResult を返す。

    COCO-WholeBody 133点インデックス:
      0-16:   体 (17点)
      17-22:  足先 (6点, foot)
      23-90:  顔 (68点)
      91-111: 左手 (21点)
      112-132: 右手 (21点)
    """

    def __init__(self, device: str = "cpu", score_threshold: float = 0.3):
        """
        Args:
            device: "cpu" or "cuda:0"
            score_threshold: この値未満のキーポイントは無効とみなす
        """
        self.device = device
        self.score_threshold = score_threshold
        self._inferencer = None

    def _load(self) -> None:
        """MMPoseInferencer を遅延ロードする（初回推論時に呼ばれる）"""
        if self._inferencer is not None:
            return

        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mmpose がインストールされていません。\n"
                "以下を実行してください:\n"
                "  pip install openmim\n"
                "  mim install mmengine mmcv mmdet mmpose"
            ) from e

        print(
            f"[Detector] MMPose wholebody モデルをロード中 (device={self.device}) ..."
        )
        self._inferencer = MMPoseInferencer(
            pose2d="wholebody",
            device=self.device,
        )
        print("[Detector] ロード完了")

    def infer(self, frame: np.ndarray) -> list[PoseResult]:
        """
        1フレームから全人物のキーポイントを検出する。

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            検出された人物ごとの PoseResult リスト（空の場合もある）
        """
        self._load()

        # MMPoseInferencer は numpy 配列を直接受け取れる
        results_gen = self._inferencer(
            frame,
            show=False,
            return_vis=False,
        )
        results = next(results_gen)

        pose_results: list[PoseResult] = []
        predictions = results.get("predictions", [[]])[0]

        for pred in predictions:
            kps = np.array(pred["keypoints"], dtype=np.float32)  # (133, 2)
            scores = np.array(pred["keypoint_scores"], dtype=np.float32)  # (133,)
            bbox = np.array(pred.get("bbox", [[0, 0, 0, 0]])[0], dtype=np.float32)

            # スコアが閾値未満の点は負の座標にマスク（三角測量で除外される）
            mask = scores < self.score_threshold
            kps[mask] = -1.0

            pose_results.append(PoseResult(keypoints=kps, scores=scores, bbox=bbox))

        return pose_results
