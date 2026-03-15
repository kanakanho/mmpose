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
import queue
import threading

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

    OSCアドレス文字列を初期化時にキャッシュし、送信を専用バックグラウンド
    スレッドに分離することでメインループをブロックしない。

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

        # OSCアドレス文字列をキャッシュ（毎フレームのf-string生成を排除）
        self._body_addrs = [f"/body/{name}" for name in BODY_KEYPOINT_NAMES]
        self._left_addrs = [f"/hand/left/{name}" for name in HAND_KEYPOINT_NAMES]
        self._right_addrs = [f"/hand/right/{name}" for name in HAND_KEYPOINT_NAMES]
        self._face_addrs = [f"/face/{i}" for i in range(68)]

        # 非同期送信キュー（maxsize=1でフレームドロップ方式、古いデータを捨てる）
        self._send_queue: queue.Queue = queue.Queue(maxsize=1)
        self._send_thread = threading.Thread(
            target=self._send_worker, daemon=True, name="OSCSendThread"
        )
        self._send_thread.start()

        print(f"[OSC] 送信先: {host}:{port} (非同期送信スレッド起動済み)")

    # ── 内部送信メソッド ──────────────────────────────────────────────────────

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

    def _send_worker(self) -> None:
        """バックグラウンドスレッド: キューからSplitPoseを受け取って送信する。"""
        while True:
            item = self._send_queue.get()
            if item is None:
                break
            kind, payload = item
            if kind == "pose":
                self._do_send_pose(payload)
            elif kind == "frame":
                self._client.send_message("/frame", int(payload))

    def _do_send_pose(self, split: SplitPose) -> None:
        """キャッシュ済みアドレスを使って128点を送信する（ワーカースレッド内）。"""
        for i, addr in enumerate(self._body_addrs):
            self._send_xyz(addr, split.body[i])
        for i, addr in enumerate(self._left_addrs):
            self._send_xyz(addr, split.left_hand[i])
        for i, addr in enumerate(self._right_addrs):
            self._send_xyz(addr, split.right_hand[i])
        for i, addr in enumerate(self._face_addrs):
            self._send_xyz(addr, split.face[i])

    # ── 公開API ───────────────────────────────────────────────────────────────

    def send_pose(self, split: SplitPose) -> None:
        """
        SplitPose の全座標をOSCで送信する（非同期キューに積む）。

        キューが満杯の場合は古いアイテムを捨てて最新フレームを優先する。
        """
        # 古いフレームを捨てて新しいものだけキューに入れる
        try:
            self._send_queue.get_nowait()
        except queue.Empty:
            pass
        self._send_queue.put_nowait(("pose", split))

    def send_frame_start(self, frame_idx: int) -> None:
        """フレーム開始を通知する（同期用）"""
        self._client.send_message("/frame", int(frame_idx))
