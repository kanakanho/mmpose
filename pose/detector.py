"""
MMPose 推論モジュール

RTMPose wholebody (COCO-WholeBody 133点) により
体・顔・両手のキーポイントを各カメラフレームから検出する。

Apple Silicon: device="mps"、NVIDIA GPU: device="cuda:0"、非対応時: "cpu" に自動フォールバック。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

import numpy as np


@dataclass
class PoseResult:
    """1人分の推論結果"""

    keypoints: np.ndarray  # shape (133, 2) - 2D座標 [x, y]
    scores: np.ndarray  # shape (133,)   - 信頼度スコア
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))  # [x1, y1, x2, y2]


class WholebodyDetector:
    # MMPoseのグローバルレジストリ初期化は並列実行非対応のため、クラスレベルでロックする
    _load_lock = threading.Lock()

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

        with WholebodyDetector._load_lock:
            # ロック取得後に再チェック（他スレッドがロード済みの場合をスキップ）
            if self._inferencer is not None:
                return

            self._load_inner()

    def _load_inner(self) -> None:
        """実際のロード処理（_load_lock 保持中に呼ばれること）"""
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mmpose がインストールされていません。\n"
                "以下を実行してください:\n"
                "  pip install openmim\n"
                "  mim install mmengine mmcv mmdet mmpose"
            ) from e

        # デバイス可用性チェック・自動フォールバック
        # NOTE: mmcv のカスタム NMS 拡張 (mmcv/ops/nms.py) は MPS 未対応のため、
        #       Apple Silicon でも device="mps" は CPU にフォールバックする。
        device = self.device
        if device == "mps":
            print(
                "[Detector] mmcv NMS が MPS 未対応のため CPU にフォールバックします。"
            )
            device = "cpu"
        elif device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    print(
                        f"[Detector] {device} が利用できません。CPU にフォールバックします。"
                    )
                    device = "cpu"
            except Exception:
                device = "cpu"

        self.device = device
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
