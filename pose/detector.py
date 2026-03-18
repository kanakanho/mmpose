"""
MMPose 推論モジュール

MMPose の topdown wholebody (COCO-WholeBody 133点) モデルにより
体・顔・両手のキーポイントを各カメラフレームから検出する。

Apple Silicon: device="mps"、NVIDIA GPU: device="cuda:0"、非対応時: "cpu" に自動フォールバック。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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

    _MODEL_PRESETS = {
        # 高精度（デフォルト）
        "high": {
            "config": "td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288.py",
            "checkpoint": (
                "https://download.openmmlab.com/mmpose/top_down/hrnet/"
                "hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"
            ),
        },
        # 速度寄り（従来）
        "balanced": {
            "config": "td-hm_res50_8xb64-210e_coco-wholebody-256x192.py",
            "checkpoint": (
                "https://download.openmmlab.com/mmpose/top_down/resnet/"
                "res50_coco_wholebody_256x192-9e37ed88_20201004.pth"
            ),
        },
    }

    """
    MMPose topdown API の wholebody モデルをラップし、
    各フレームから PoseResult を返す。

    COCO-WholeBody 133点インデックス:
      0-16:   体 (17点)
      17-22:  足先 (6点, foot)
      23-90:  顔 (68点)
      91-111: 左手 (21点)
      112-132: 右手 (21点)
    """

    def __init__(self, device: str = "cpu", score_threshold: float = 0.3, model_preset: str = "high"):
        """
        Args:
            device: "cpu" or "cuda:0"
            score_threshold: この値未満のキーポイントは無効とみなす
            model_preset: "high" (HRNet-W48 384x288) または "balanced" (ResNet-50 256x192)
        """
        self.device = device
        self.score_threshold = score_threshold
        self.model_preset = model_preset.strip().lower()
        self._model = None
        self._inference_topdown = None

    def _load(self) -> None:
        """推論モデルを遅延ロードする（初回推論時に呼ばれる）"""
        if self._model is not None:
            return

        with WholebodyDetector._load_lock:
            # ロック取得後に再チェック（他スレッドがロード済みの場合をスキップ）
            if self._model is not None:
                return

            self._load_inner()

    def _load_inner(self) -> None:
        """実際のロード処理（_load_lock 保持中に呼ばれること）"""
        try:
            import mmpose  # type: ignore
            from mmpose.apis import inference_topdown, init_model  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mmpose がインストールされていません。\n"
                "以下を実行してください:\n"
                "  pip install mmcv-lite==2.0.1 mmpose==1.0.0"
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
        print(f"[Detector] device={device}")

        # mmdet/mmcv の C++ 拡張依存を避けるため、MMPose の topdown API を直接利用する。
        if self.model_preset not in self._MODEL_PRESETS:
            print(
                f"[Detector] 未知のモデルプリセット='{self.model_preset}'。"
                "'high' を使用します。"
            )
            self.model_preset = "high"

        preset = self._MODEL_PRESETS[self.model_preset]
        mmpose_root = Path(mmpose.__file__).resolve().parent
        config_path = (
            mmpose_root
            / ".mim"
            / "configs"
            / "wholebody_2d_keypoint"
            / "topdown_heatmap"
            / "coco-wholebody"
            / preset["config"]
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"MMPose config が見つかりません: {config_path}"
            )

        checkpoint_url = preset["checkpoint"]

        print(
            "[Detector] MMPose wholebody モデルをロード中 "
            f"(preset={self.model_preset}, device={self.device}) ..."
        )
        self._model = init_model(
            str(config_path),
            checkpoint=checkpoint_url,
            device=self.device,
        )
        self._inference_topdown = inference_topdown
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

        pose_results: list[PoseResult] = []
        results = self._inference_topdown(self._model, frame, bboxes=None)

        h, w = frame.shape[:2]
        for data_sample in results:
            pred_instances = data_sample.pred_instances
            if not hasattr(pred_instances, "keypoints") or not hasattr(
                pred_instances, "keypoint_scores"
            ):
                continue

            all_kps = np.asarray(pred_instances.keypoints, dtype=np.float32)
            all_scores = np.asarray(pred_instances.keypoint_scores, dtype=np.float32)
            all_bboxes = (
                np.asarray(pred_instances.bboxes, dtype=np.float32)
                if hasattr(pred_instances, "bboxes")
                else None
            )

            # 推論結果は (N, 133, 2) / (N, 133) を想定。
            if all_kps.ndim == 2:
                all_kps = all_kps[None, ...]
            if all_scores.ndim == 1:
                all_scores = all_scores[None, ...]

            num_person = min(len(all_kps), len(all_scores))
            for i in range(num_person):
                kps = all_kps[i]
                scores = all_scores[i]
                if kps.shape[0] != 133 or scores.shape[0] != 133:
                    continue

                if all_bboxes is not None and len(all_bboxes) > i:
                    bbox = all_bboxes[i]
                else:
                    bbox = np.array([0, 0, w, h], dtype=np.float32)

                # スコアが閾値未満の点は負の座標にマスク（三角測量で除外される）
                kps = kps.copy()
                mask = scores < self.score_threshold
                kps[mask] = -1.0

                pose_results.append(PoseResult(keypoints=kps, scores=scores, bbox=bbox))

        return pose_results
