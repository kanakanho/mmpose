"""
OSC 送信モジュール

python-osc を使って3D骨格/手/顔座標をリアルタイムで配信する。

OSCアドレス仕様:
  /body/{joint_name}          -> float x, float y, float z
  /hand/left/{joint_name}     -> float x, float y, float z
  /hand/right/{joint_name}    -> float x, float y, float z
  /face/{index}               -> float x, float y, float z
"""

from __future__ import annotations

import math

import numpy as np

try:
    from pythonosc.udp_client import SimpleUDPClient  # type: ignore
except ImportError as e:
    raise ImportError(
        "python-osc がインストールされていません。\n" "  pip install python-osc"
    ) from e

from pose.hand_splitter import (
    BODY_KEYPOINT_NAMES,
    HAND_KEYPOINT_NAMES,
    SplitPose,
)


class OSCSender:
    """
    SplitPose の3D座標を OSC でブロードキャストする。

    Args:
        host: 送信先ホスト
        port: 送信先ポート
        coordinate_scale: 座標のスケール係数（mmpose出力はmm単位が多い、必要に応じてm換算等）
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        coordinate_scale: float = 0.001,  # mm → m
    ):
        self.host = host
        self.port = port
        self.scale = coordinate_scale
        self._client = SimpleUDPClient(host, port)
        print(f"[OSC] 送信先: {host}:{port}")

    def _send_xyz(self, address: str, pos: np.ndarray) -> None:
        """1点の座標を送信。NaNは0として送信する。"""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            x, y, z = 0.0, 0.0, 0.0
        else:
            x *= self.scale
            y *= self.scale
            z *= self.scale
        self._client.send_message(address, [x, y, z])

    def send_pose(self, split: SplitPose) -> None:
        """
        SplitPose の全座標を OSC で送信する。

        各ジョイントのアドレス例:
          /body/left_shoulder
          /hand/left/wrist
          /hand/right/index_tip
          /face/0 ～ /face/67
        """
        # ── 体 ──────────────────────────────────────────────────────────────
        for i, name in enumerate(BODY_KEYPOINT_NAMES):
            self._send_xyz(f"/body/{name}", split.body[i])

        # ── 左手 ─────────────────────────────────────────────────────────────
        for i, name in enumerate(HAND_KEYPOINT_NAMES):
            self._send_xyz(f"/hand/left/{name}", split.left_hand[i])

        # ── 右手 ─────────────────────────────────────────────────────────────
        for i, name in enumerate(HAND_KEYPOINT_NAMES):
            self._send_xyz(f"/hand/right/{name}", split.right_hand[i])

        # ── 顔 (68点) ────────────────────────────────────────────────────────
        for i in range(len(split.face)):
            self._send_xyz(f"/face/{i}", split.face[i])

    def send_frame_start(self, frame_idx: int) -> None:
        """フレーム開始を通知する（同期用）"""
        self._client.send_message("/frame", int(frame_idx))
