"""
キャリブレーション用チェッカーボード生成スクリプト

デフォルトのキャリブレーション設定 (--board-size 9 6) に対応する
10×7 マス（内角 9×6）のチェッカーボードを生成します。

出力: checkerboard_10x7.png
  - A4 用紙（297×210mm）に収まるよう余白付きで生成
  - 印刷後、1マスが約 25mm になるよう調整済み
"""

import cv2
import numpy as np

# ── 設定 ──────────────────────────────────────────────────────────────────
COLS = 10  # 横マス数（内角 9 = COLS-1）
ROWS = 7  # 縦マス数（内角 6 = ROWS-1）
SQ_PX = 80  # 1マスのピクセルサイズ
MARGIN = 40  # 余白（px）

# ── 生成 ──────────────────────────────────────────────────────────────────
h = ROWS * SQ_PX + 2 * MARGIN
w = COLS * SQ_PX + 2 * MARGIN
board = np.ones((h, w), dtype=np.uint8) * 255  # 白背景

for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            y0 = MARGIN + r * SQ_PX
            x0 = MARGIN + c * SQ_PX
            board[y0 : y0 + SQ_PX, x0 : x0 + SQ_PX] = 0

out_path = "checkerboard_10x7.png"
cv2.imwrite(out_path, board)
print(f"生成完了: {out_path}  ({w}x{h}px, {COLS}x{ROWS}マス, 内角{COLS-1}x{ROWS-1})")
print("このファイルを印刷して使用してください。")
print(
    f"キャリブレーション実行: python main.py --mode calibrate --cameras 0 1 --board-size {COLS-1} {ROWS-1}"
)
