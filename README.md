# mmpose-multicam

MMPose を使ったマルチカメラ全身 3D 検出システムです。  
複数の物理カメラから人体・両手・顔ランドマークを検出し、OSC でリアルタイム配信します。

## 機能

| 機能                     | 内容                              |
| ------------------------ | --------------------------------- |
| **3D 人体骨格検出**      | 体 17 点 (COCO) + 足先 6 点       |
| **3D 手検出**            | 左手・右手それぞれ 21 点          |
| **顔ランドマーク検出**   | 顔 68 点                          |
| **マルチカメラ 3D 復元** | 多視点三角測量（DLT 法）          |
| **OSC 配信**             | 全身 133 点をリアルタイム送信     |
| **プレビュー**           | OpenCV による骨格オーバーレイ表示 |

---

# 環境構築

## 1. uv

```sh
$ uv venv
```

```sh
$ uv sync
```

## 2. mmcv / mmpose 2. 依存関係のインストール

```sh
$ pip install wheel setuptools==65.0.0
```

> [!NOTE]
> pip が原因で setuptools を古くしないと動かない  
> https://github.com/open-mmlab/mmcv/issues/3325

```sh
$ pip install mmcv --no-build-isolation
```

```sh
$ pip install cython
```

```sh
$ pip install mmpose --no-build-isolation
```

---

# 使い方

## Step 1: キャリブレーション（マルチカメラ 3D 使用時のみ）

チェッカーボードを用意し、全カメラに映した状態でキャプチャします。

```sh
python main.py --mode calibrate --cameras 0 1
```

- キャリブレーションウィンドウが開きます
- チェッカーボード（デフォルト: 9×6 の内角）を全カメラに映してください
- `c` キーで撮影（20 枚撮ると自動終了）
- `q` キーで中断
- 完了すると `camera_params.json` が生成されます

**キャリブレーション オプション:**

| 引数                     | デフォルト           | 説明                     |
| ------------------------ | -------------------- | ------------------------ |
| `--board-size COLS ROWS` | `9 6`                | チェッカーボードの内角数 |
| `--square-size MM`       | `25.0`               | 1 マスのサイズ（mm）     |
| `--calib-frames N`       | `20`                 | 収集するフレーム数       |
| `--params-path PATH`     | `camera_params.json` | パラメータの保存先       |

## Step 2: 推論実行

```sh
# マルチカメラ 3D モード（キャリブレーション済み）
python main.py --mode run --cameras 0 1 --osc-host 127.0.0.1 --osc-port 9000

# 3 台以上のカメラ
python main.py --mode run --cameras 0 1 2 --osc-port 9000

# 単体カメラ（キャリブレーション不要・2D 出力）
python main.py --mode run --cameras 0 --no-calibration

# プレビューなし（ヘッドレス動作）
python main.py --mode run --cameras 0 1 --no-preview
```

**ESC キーで終了します。**

### 全オプション一覧

| 引数                | デフォルト           | 説明                                |
| ------------------- | -------------------- | ----------------------------------- |
| `--mode`            | `run`                | `calibrate` または `run`            |
| `--cameras IDX...`  | `0`                  | カメラインデックス（複数指定可）    |
| `--no-calibration`  | —                    | キャリブレーションなし（2D モード） |
| `--score-threshold` | `0.3`                | キーポイント信頼度の閾値            |
| `--device`          | `cpu`                | 推論デバイス（`cpu` / `cuda:0`）    |
| `--osc-host`        | `127.0.0.1`          | OSC 送信先ホスト                    |
| `--osc-port`        | `9000`               | OSC 送信先ポート                    |
| `--osc-scale`       | `0.001`              | 座標スケール係数（mm→m）            |
| `--no-preview`      | —                    | OpenCV プレビューを無効化           |
| `--preview-width`   | `640`                | プレビュー表示幅（px）              |
| `--params-path`     | `camera_params.json` | カメラパラメータのパス              |

---

# OSC アドレス仕様

座標は `(x, y, z)` の 3 つの float 値で送信されます（単位: m）。

## フレーム同期

| アドレス | 型    | 説明         |
| -------- | ----- | ------------ |
| `/frame` | `int` | フレーム番号 |

## 体（17 点）

`/body/{joint_name}` → `float x, float y, float z`

| ジョイント名                       | 部位 |
| ---------------------------------- | ---- |
| `nose`                             | 鼻   |
| `left_eye` / `right_eye`           | 目   |
| `left_ear` / `right_ear`           | 耳   |
| `left_shoulder` / `right_shoulder` | 肩   |
| `left_elbow` / `right_elbow`       | 肘   |
| `left_wrist` / `right_wrist`       | 手首 |
| `left_hip` / `right_hip`           | 腰   |
| `left_knee` / `right_knee`         | 膝   |
| `left_ankle` / `right_ankle`       | 足首 |

## 手（各 21 点）

`/hand/left/{joint_name}` または `/hand/right/{joint_name}` → `float x, float y, float z`

| ジョイント名                                              | 部位     |
| --------------------------------------------------------- | -------- |
| `wrist`                                                   | 手首     |
| `thumb_cmc` / `thumb_mcp` / `thumb_ip` / `thumb_tip`      | 親指     |
| `index_mcp` / `index_pip` / `index_dip` / `index_tip`     | 人差し指 |
| `middle_mcp` / `middle_pip` / `middle_dip` / `middle_tip` | 中指     |
| `ring_mcp` / `ring_pip` / `ring_dip` / `ring_tip`         | 薬指     |
| `pinky_mcp` / `pinky_pip` / `pinky_dip` / `pinky_tip`     | 小指     |

## 顔（68 点）

`/face/{0-67}` → `float x, float y, float z`

---

# ファイル構成

```
main.py                    # エントリーポイント
calibration/
  calibrate.py             # チェッカーボードキャリブレーション
pose/
  detector.py              # MMPoseInferencer ラッパー（wholebody 133 点）
  triangulate.py           # DLT 法による多視点三角測量
  hand_splitter.py         # 133 点から体・顔・手に分離
output/
  osc_sender.py            # python-osc による OSC 配信
  preview.py               # OpenCV マルチカメラプレビュー
camera_params.json         # キャリブレーション結果（生成されます）
```

---

# 技術仕様

- **検出モデル:** RTMPose wholebody（COCO-WholeBody 133 点）
- **3D 復元:** DLT（Direct Linear Transform）法による多視点三角測量
- **座標系:** カメラ 0 を基準座標系として統一
- **信頼度:** スコアが閾値未満の点は NaN として三角測量から除外
