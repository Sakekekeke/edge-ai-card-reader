# プロジェクト構成 & 引き継ぎ資料

新しいチャットの冒頭にこのファイルの内容を貼ってください。
Claudeはこの資料をもとにプロジェクトの全体像を把握します。

最終更新: 2026/04/11

---

<!--
## この資料に求められる内容（定義）

1. 環境情報: 開発に必要なPC・ラズパイのスペック、OS、接続方法、Python環境
2. 開発フロー: Git運用、ファイル転送方法、スクリプトの実行手順
3. ファイル構成: PC側・ラズパイ側の全ファイルとその役割
4. 実験結果: モデル精度・FPS等の定量データ（再学習判断の根拠になる）
5. 採用技術・判断ログ: 何を選び何を捨てたか（同じ検討を繰り返さないため）
6. 進捗管理: 完了済み・作業中・次にやることの明確な区分
7. 技術的注意事項: ハマりポイントや環境固有の制約
8. 変更履歴: いつ何が変わったかの記録
-->

## 環境情報

### PC (Windows)
- GPU: RTX 4060 Ti
- 作業ディレクトリ: `D:\edge-ai-card-reader\`
- Python: venv (`D:\edge-ai-card-reader\.venv\`)

### ラズパイ (Raspberry Pi 5, 8GB)
- OS: Debian Trixie (64bit)
- SSH: `ssh fujiwara@192.168.141.207` または `ssh fujiwara@ml-dev-pi.local`
- 作業ディレクトリ: `~/edge-ai-card-reader/`
- Python: venv（`source venv/bin/activate` してからスクリプト実行）
- カメラ: picamera2で動作確認済み

### ラズパイ側パッケージ状況
- インストール済み: ultralytics, picamera2, opencv-python, numpy
- 未インストール: flask（live_stream.py実行前に `pip install flask` が必要）
- ※ pip installする際は `--break-system-packages` フラグが必要

---

## 開発フロー

- Git管理・pushはPC側で行う
- ラズパイへのファイル転送はSCP:
  ```
  scp scripts/live_stream.py fujiwara@192.168.141.207:~/edge-ai-card-reader/scripts/
  ```
- ラズパイ側でgit pullする運用は未確立（今後整備予定）
- ラズパイ上でのスクリプト実行手順:
  ```
  cd ~/edge-ai-card-reader
  source venv/bin/activate
  python scripts/スクリプト名.py
  ```

---

## ファイル構成

### PC (Windows)

```
D:\edge-ai-card-reader\
├── .gitignore
├── LICENSE
├── README.md
├── PROJECT_STRUCTURE.md            # この引き継ぎ資料
├── check_augmentation.py           # Augmentation可視化スクリプト
├── scripts/
│   ├── train_config.py             # 学習の共通設定（パラメータ・Augmentation）
│   ├── train_exp_001.py            # 実験001: YOLO11n (imgsz=640)
│   ├── train_exp_002.py            # 実験002: YOLO26n (imgsz=640)
│   ├── benchmark_raspi.py          # ラズパイFPS計測スクリプト
│   ├── export_and_benchmark.py     # フォーマット変換＋FPS計測
│   └── live_stream.py              # リアルタイム検出MJPEG配信サーバー
├── docs/
│   └── YOLO_OUTPUT_GUIDE.md        # YOLO出力ファイル解説マニュアル
├── dataset/                        # .gitignore対象
│   ├── train/images/, train/labels/
│   ├── valid/images/, valid/labels/
│   ├── test/images/, test/labels/
│   └── data.yaml
├── runs/                           # .gitignore対象（学習結果）
│   └── detect/runs/
│       ├── exp_001_yolo11n/        # 実験001の全結果
│       └── exp_002_yolo26n2/       # 実験002の全結果
├── aug_preview/                    # .gitignore対象（Augmentation確認画像）
└── .venv/                          # .gitignore対象（Python仮想環境）
```

### ラズパイ (Raspberry Pi 5)

```
~/edge-ai-card-reader/
├── models/
│   ├── exp_001_best.pt             # 実験001 PyTorch形式
│   ├── exp_001_best.onnx           # 実験001 ONNX形式
│   ├── exp_001_best_ncnn_model/    # 実験001 NCNN形式（31.1FPS）★採用モデル
│   ├── exp_002_best.pt             # 実験002 PyTorch形式
│   └── exp_002_best.onnx           # 実験002 ONNX形式
├── scripts/
│   ├── benchmark_raspi.py          # モデル比較FPS計測
│   ├── export_and_benchmark.py     # フォーマット変換＋FPS計測
│   └── live_stream.py              # リアルタイム検出MJPEG配信サーバー
└── venv/                           # Python仮想環境
```

---

## GitHub

- リポジトリ: https://github.com/Sakekekeke/-edge-ai-card-reader
- ブランチ: main

---

## 実験結果サマリー

| 実験 | モデル | mAP50 | mAP50-95 | Precision | Recall |
|------|--------|-------|----------|-----------|--------|
| exp_001 | YOLO11n | 0.9950 | 0.9766 | 0.9985 | 0.9995 |
| exp_002 | YOLO26n | 0.9945 | 0.9675 | 0.9861 | 0.9905 |

| モデル + 形式 | ラズパイFPS |
|---------------|-----------|
| YOLO11n PyTorch | 3.5 |
| YOLO11n ONNX | 6.5 |
| YOLO11n NCNN | 31.1 |
| YOLO26n PyTorch | 3.7 |
| YOLO26n ONNX | 7.4 |
| YOLO26n NCNN | 非対応 |

---

## 採用技術・判断ログ

- **採用モデル**: YOLO11n NCNN形式（31.1FPS、mAP50=0.995）
  - パス: `~/edge-ai-card-reader/models/exp_001_best_ncnn_model/`
- **YOLO26n不採用理由**: NCNN変換非対応（ultralytics 8.4.x時点）、精度もYOLO11nより若干低い
- **NCNN採用理由**: PyTorch(3.5FPS)→ONNX(6.5FPS)→NCNN(31.1FPS)と圧倒的に高速。目標30FPS超え達成
- **MJPEG配信採用理由**: SSH X11転送では5-10FPS程度に低下するため、JPEG圧縮+HTTP配信方式を採用。ブラウザで閲覧可能でデモにも適する
- **学習データ**: Roboflowのトランプカードデータセット（52クラス）。正面・均一照明が中心で、手持ち・斜めの実環境データは未収集

---

## 進捗・フェーズ管理

### 完了済み
- **フェーズ1**: ラズパイ5環境構築（picamera2でカメラ動作確認済み）
- **フェーズ2**: モデル学習 & ラズパイ最適化
  - 実験001: YOLO11n学習完了
  - 実験002: YOLO26n学習完了
  - FPS計測: YOLO11n NCNN形式で31.1FPS達成（目標30FPS超え）
  - YOLO26nはNCNN変換非対応と判明
  - GitHubにスクリプト・ドキュメント・README整備済み
  - ラズパイ側ファイル整理完了（scripts/フォルダに統合、不要ファイル削除）

### 作業中
- リアルタイム検出MJPEG配信（live_stream.py作成済み、ラズパイで動作確認待ち）

### 次にやること
- **フェーズ3**: 自前データ撮影（手持ち・斜めカード）→ 再学習で精度改善
- **フェーズ4**: ブラックジャック戦略テーブル実装
- **フェーズ5**: リアルタイムUI（HIT/STAND表示）+ デモ動画

---

## 技術的注意事項

- picamera2のフレームはRGBA形式 → `cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)` が必要
- MJPEG配信はFlask使用、ポート8080、ブラウザで `http://192.168.141.207:8080` でアクセス
- ラズパイ側で `cd ~/edge-ai-card-reader` してから実行する前提（MODEL_PATHが相対パス）
- NCNN形式のモデルはフォルダごと必要（`exp_001_best_ncnn_model/` 配下に複数ファイル）

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026/04/10 | PROJECT_STRUCTURE.md 初版作成 |
| 2026/04/11 | live_stream.py追加、ラズパイ側ファイル整理（scripts/統合・不要ファイル削除）、引き継ぎ資料として内容拡充（環境情報・開発フロー・判断ログ・変更履歴を追加） |
