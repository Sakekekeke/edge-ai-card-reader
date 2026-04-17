# プロジェクト構成 & 引き継ぎ資料

新しいチャットの冒頭にこのファイルの内容を貼ってください。
Claudeはこの資料をもとにプロジェクトの全体像を把握します。

最終更新: 2026/04/12

---

## Claudeへの行動ルール

1. この資料の内容に少しでも不明点・曖昧な点がある場合、推測せず必ずユーザーに質問すること
2. 質問して得られた回答は、該当するセクションまたは「未解決・確認済みの疑問」セクションに追記すること
3. ファイルを新規作成・修正した場合は「ファイル構成」と「変更履歴」の更新内容をユーザーに提示すること
4. 技術的な判断をした場合は「採用技術・判断ログ」に根拠とともに記録すること
5. この資料はチャットをまたいだ唯一の引き継ぎ手段であるため、情報の欠落・曖昧さは許容しない
6. 資料が500行を超えたら、完了済みの実験詳細や解決済みQ&Aを `docs/ARCHIVE.md` に移すことを提案すること
7. フェーズ3に着手したら「作業中」セクションを進捗管理に追加すること
8. 変更履歴は1行1〜3項目を目安にし、肥大化したらARCHIVE.mdに移すこと
9. 変更履歴を追記する際は、冒頭の「最終更新」日付も合わせて更新すること

## この資料に求められる内容（定義）

1. プロジェクト概要: 何を作っているか、最終ゴール
2. 環境情報: 開発に必要なPC・ラズパイのスペック、OS、接続方法、Python環境
3. 開発フロー: Git運用、ファイル転送方法、スクリプトの実行手順
4. ファイル構成: PC側・ラズパイ側の全ファイルとその役割
5. 実験結果: モデル精度・FPS等の定量データ（再学習判断の根拠になる）
6. 採用技術・判断ログ: 何を選び何を捨てたか（同じ検討を繰り返さないため）
7. 進捗管理: 完了済み・作業中・次にやることの明確な区分
8. 技術的注意事項: ハマりポイントや環境固有の制約
9. 未解決・確認済みの疑問: Claudeが質問し回答を得た内容の蓄積
10. 変更履歴: いつ何が変わったかの記録

---

## プロジェクト概要

Raspberry Pi 5 + カメラでトランプカードをリアルタイム検出し、カードゲームの戦略支援を行うエッジAIシステムを開発している。最初の実装対象としてブラックジャック（HIT/STAND判定）を選択。将来的にはポーカーのハンド判定など他のトランプゲームへの応用も視野に入れている。

**最終ゴール**: カメラに映ったカードを認識 → ゲームルールに基づく最適行動をリアルタイム表示

---

## 環境情報

### PC (Windows)
- GPU: RTX 4060 Ti（学習時に使用。train_config.pyで`device=0`指定でCUDA利用）
- 作業ディレクトリ: `D:\edge-ai-card-reader\`
- Python: 3.13.2, venv (`D:\edge-ai-card-reader\.venv\`)
- エディタ: VSCode（学習スクリプト等はVSCodeから直接実行可）
- 主要パッケージ: ultralytics 8.4.36, PyTorch 2.7.1+cu118
- 学習時間目安: 1実験あたり1〜2時間程度（50エポック、RTX 4060 Ti）

### ラズパイ (Raspberry Pi 5, 8GB)
- OS: Debian Trixie (64bit)
- SSH: `ssh fujiwara@ml-dev-pi.local`（優先）。mDNSが効かない場合は `ssh fujiwara@192.168.141.207`
- 作業ディレクトリ: `~/edge-ai-card-reader/`
- Python: 3.13.5（システムにこの1バージョンのみ。`python3`で起動）
- venv: `source venv/bin/activate` してからスクリプト実行
- カメラ: picamera2で動作確認済み（IMX708センサー）

### 主要パッケージバージョン（ラズパイ）
- ultralytics: 8.4.33
- PyTorch: 2.11.0+cpu
- ncnn: 1.0.20260114
- パッケージ詳細は requirements-raspi.txt を参照
- パッケージ配置:
  - apt経由（システム）: picamera2, numpy, pillow, PyYAML, flask
  - venv内（pip）: torch, torchvision, ncnn
  - グローバル pip（`~/.local/lib/`）: ultralytics, matplotlib, scipy等
  - venvは `--system-site-packages` で作成し、上記全てを参照
- ultralyticsのバージョン差（PC:8.4.36 / ラズパイ:8.4.33）はインストール時期の違い。動作上問題ないが、PC側で学習→エクスポートしたモデルがラズパイ側で読めない可能性がゼロではない。フェーズ3の再学習時にバージョンを揃えることを検討

---

## 開発フロー

- Git管理・pushはPC側で行う（ブランチ: main）
- ラズパイへのファイル転送はSCP:
  ```
  # スクリプト（単一ファイル）
  scp scripts/live_stream.py fujiwara@ml-dev-pi.local:~/edge-ai-card-reader/scripts/
  # NCNNモデル（フォルダごと転送が必要）
  scp -r models/exp_001_best_ncnn_model fujiwara@ml-dev-pi.local:~/edge-ai-card-reader/models/
  ```
- ラズパイ側でgit pullする運用は未確立（今後整備予定）
- ラズパイ上でのスクリプト実行手順:
  ```
  cd ~/edge-ai-card-reader
  source venv/bin/activate
  python scripts/スクリプト名.py
  ```
- PC側の学習スクリプトはVSCode実行・コマンドライン実行のどちらも可
  （sys.path修正済みのため実行方法を問わずtrain_config.pyのimportが動作する）

### デモ実行手順（MJPEG配信）
```
# ラズパイ側
cd ~/edge-ai-card-reader
source venv/bin/activate
python scripts/live_stream.py

# PC側ブラウザでアクセス
http://ml-dev-pi.local:8080
# mDNSが効かない場合: http://192.168.141.207:8080
```

### live_stream.py 主要パラメータ一覧

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| MODEL_PATH | `models/exp_001_best_ncnn_model` | NCNNフォルダ指定 |
| IMGSZ | 416 | NCNNエクスポート時サイズに合わせる（640不可） |
| CAMERA_SIZE | 640x480 | picamera2の取得解像度 |
| JPEG_QUALITY | 80 | MJPEG配信のJPEG圧縮品質 |
| SERVER_PORT | 8080 | Flask配信ポート |
| FPS_WINDOW | 30 | FPS計算用の直近フレーム数 |
| Flask host | 0.0.0.0 | LAN内全端末からアクセス可能 |

### venv再構築手順（環境が壊れた場合）

**PC側:**
```
cd D:\edge-ai-card-reader
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-pc.txt
```

**ラズパイ側:**
```
cd ~/edge-ai-card-reader
# picamera2はapt経由で事前にインストールが必要
# sudo apt install python3-picamera2
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-raspi.txt
```
注意: `--system-site-packages`はaptでインストールしたpicamera2をvenvから参照するために必要。

---

## ファイル構成

### PC (Windows)

```
D:\edge-ai-card-reader\
├── .gitignore                      # 除外対象: dataset/, aug_preview/, *.pt, *.onnx, runs/, .env, __pycache__/
├── LICENSE                         # MIT License (Copyright 2026 ryohei fujiwara)
├── README.md
├── PROJECT_STRUCTURE.md            # この引き継ぎ資料
├── requirements-pc.txt             # PC側パッケージ一覧（学習・開発用）
├── requirements-raspi.txt          # ラズパイ側パッケージ一覧（推論・配信用。venv再構築時にSCP転送）
├── scripts/
│   ├── train_config.py             # 学習の共通設定（パラメータ・Augmentation）
│   ├── train_exp_001.py            # 実験001: YOLO11n (imgsz=640)
│   ├── train_exp_002.py            # 実験002: YOLO26n (imgsz=640)
│   ├── check_augmentation.py       # Augmentation可視化スクリプト
│   ├── benchmark_raspi.py          # ラズパイ用: PyTorch形式のFPS計測（モデル比較用）
│   ├── export_and_benchmark.py     # ラズパイ用: PyTorch→ONNX→NCNN変換＋各形式のFPS計測
│   ├── compare_imgsz.py            # ラズパイ用: imgsz 416 vs 640 の精度・FPS比較（完了済み。再比較用に保持）
│   └── live_stream.py              # ラズパイ用: リアルタイム検出MJPEG配信サーバー
├── docs/
│   └── YOLO_OUTPUT_GUIDE.md        # YOLO出力ファイル解説マニュアル
├── dataset/                        # .gitignore対象
│   ├── train/images/, train/labels/
│   ├── valid/images/, valid/labels/
│   ├── test/images/, test/labels/
│   └── data.yaml
├── runs/                           # .gitignore対象（学習結果）
│   └── detect/
│       ├── exp_001_yolo11n/        # 実験001の全結果
│       ├── exp_002_yolo26n/        # 実験002の全結果
│       └── val/                    # 検証結果
├── aug_preview/                    # .gitignore対象（Augmentation確認画像）
└── .venv/                          # .gitignore対象（Python仮想環境）
```
※ 「ラズパイ用」スクリプトもPC側に置いているのはGit管理を一元化するため。SCPでラズパイに転送して使用する。

### ラズパイ (Raspberry Pi 5)

```
~/edge-ai-card-reader/
├── models/
│   ├── exp_001_best.pt             # 実験001 PyTorch形式
│   ├── exp_001_best.onnx           # 実験001 ONNX形式
│   ├── exp_001_best_ncnn_model/    # ★採用モデル（imgsz=416）。live_stream.pyが参照するフォルダ
│   ├── exp_001_416_ncnn_model/     # 416版のコピー（compare_imgsz.py比較用。採用モデルと同一内容。不要になれば削除可）
│   ├── exp_001_640_ncnn_model/     # 640版（compare_imgsz.py比較用。不採用だが記録として保持）
│   ├── exp_002_best.pt             # 実験002 PyTorch形式（不採用。記録として保持、削除可）
│   └── exp_002_best.onnx           # 実験002 ONNX形式（不採用。記録として保持、削除可）
├── scripts/
│   ├── benchmark_raspi.py          # PyTorch形式のFPS計測（モデル比較用）
│   ├── export_and_benchmark.py     # PyTorch→ONNX→NCNN変換＋各形式のFPS計測
│   ├── compare_imgsz.py            # imgsz 416 vs 640 の精度・FPS比較
│   └── live_stream.py              # リアルタイム検出MJPEG配信サーバー
└── venv/                           # Python仮想環境（CPU版PyTorchで再構築済み）
```
※ ラズパイ側にdataset/は不要（推論・配信のみのため。学習はPC側で実施）

---

## GitHub

- リポジトリ: https://github.com/Sakekekeke/edge-ai-card-reader
- ブランチ: main
- ライセンス: MIT License
- データセットライセンス: Public Domain（Roboflow提供。MITと矛盾なし）

---

## データセット情報

- **出典**: Roboflow play_cards_standard v2（Public Domain）
- **枚数**: 10,100枚
  - train: 7,070枚 / valid: 2,020枚 / test: 1,010枚（約70:20:10）
- **クラス数**: 52（トランプ全種）
- **クラス名の命名規則**: `{数字/文字}{スート}` の形式
  - 数字: 2-10, J, Q, K, A
  - スート: C=♣(クラブ), D=♦(ダイヤ), H=♥(ハート), S=♠(スペード)
  - 例: 10C=10♣, JS=J♠, AD=A♦, QH=Q♥
- **data.yaml**: パスは`../train/images`等の相対パス指定。ultralyticsがdata.yamlの配置場所を基準に自動解決する（Roboflow標準の出力形式。手動変更不要）
- **特性**: 正面・均一照明が中心。手持ち・斜めの実環境データは未収集（フェーズ3で対応予定）
- **全52クラス一覧** (data.yaml準拠):
  ```
  10C, 10D, 10H, 10S, 2C, 2D, 2H, 2S, 3C, 3D, 3H, 3S,
  4C, 4D, 4H, 4S, 5C, 5D, 5H, 5S, 6C, 6D, 6H, 6S,
  7C, 7D, 7H, 7S, 8C, 8D, 8H, 8S, 9C, 9D, 9H, 9S,
  AC, AD, AH, AS, JC, JD, JH, JS, KC, KD, KH, KS, QC, QD, QH, QS
  ```
  フェーズ4でのBJ値マッピング: A→1or11, 2-10→数字通り, J/Q/K→10

---

## 学習パラメータ（要点）

詳細は `scripts/train_config.py` を参照。
train_config.pyはグローバル変数（辞書 `TRAIN_PARAMS`, `AUGMENTATION`）と関数（`download_dataset()`）を定義。
各実験スクリプトから `from train_config import AUGMENTATION, DATA_YAML, TRAIN_PARAMS, download_dataset` で呼び出す。

- imgsz=640, batch=32, epochs=50, patience=15
- optimizer=SGD, lr0=0.01, warmup_epochs=3
- Augmentation: hsv_h/s/v, degrees=10, translate=0.1, scale=0.5, mosaic=1.0
- 無効化: fliplr/flipud（カードの向きに意味あり）, shear/perspective（フェーズ3で実データ対応）, mixup（境界がぼやける）

---

## 実験結果サマリー

### 実験の採番ルール

`exp_{連番}_{変更内容}` の形式。何を変えたかが一目でわかるようにする。
- exp_001_yolo11n: ベースライン（YOLO11n）
- exp_002_yolo26n: アーキテクチャ比較（YOLO26n）
- 今後の例: exp_003_custom_data（自前データ追加）、exp_004_imgsz416（学習imgsz変更）等

### モデル学習結果

| 実験 | モデル | mAP50 | mAP50-95 | Precision | Recall |
|------|--------|-------|----------|-----------|--------|
| exp_001 | YOLO11n | 0.9950 | 0.9766 | 0.9985 | 0.9995 |
| exp_002 | YOLO26n | 0.9945 | 0.9675 | 0.9861 | 0.9905 |

### ラズパイFPS計測

| モデル + 形式 | FPS | 備考 |
|---------------|-----|------|
| YOLO11n PyTorch | 3.5 | |
| YOLO11n ONNX | 6.5 | |
| YOLO11n NCNN imgsz=416 | 30.9 | ★採用 |
| YOLO11n NCNN imgsz=640 | 13.1 | 目標30FPS未達 |
| YOLO11n NCNN imgsz=416 (MJPEG配信時) | 約27 | 描画+JPEG圧縮込み、ブラウザ1台接続、JPEG品質80 |
| YOLO26n PyTorch | 3.7 | |
| YOLO26n ONNX | 7.4 | |
| YOLO26n NCNN | — | 非対応 |

### imgsz 416 vs 640 比較

比較条件: 同一フレーム1枚での検出結果比較（カード6-7枚映した状態）。FPSは50フレーム計測の平均値。

| 項目 | imgsz=416 | imgsz=640 |
|------|-----------|-----------|
| 検出数 | 7 | 8（7Dを追加検出） |
| 高conf検出 | JS 0.96, 10D 0.94, QH 0.93 | JS 0.94, QH 0.93, 10D 0.92 |
| 低conf検出 | 3D 0.61, JC 0.56 | 3D 0.89, 7D 0.87, JS 0.58 |
| 平均FPS | 30.9 | 13.1 |

→ 640は低confidenceの検出がやや改善されるが、FPSが半分以下に低下。416で実用上十分。
※ 精度比較は1枚のスポットチェック。FPS差（2.4倍）が支配的な判断要因であり、精度はフェーズ3で自前データ追加により改善する方針のため、多フレーム統計は実施していない。

---

## 採用技術・判断ログ

- **採用モデル**: YOLO11n NCNN形式（imgsz=416）
  - パス: `~/edge-ai-card-reader/models/exp_001_best_ncnn_model/`
- **YOLO26n不採用理由**: NCNN変換非対応（ultralytics 8.4.33時点）、精度もYOLO11nより若干低い
- **NCNN採用理由**: PyTorch(3.5FPS)→ONNX(6.5FPS)→NCNN(30.9FPS)と圧倒的に高速。目標30FPS超え達成
- **imgsz=416採用理由**: 学習はimgsz=640だが、416と640を実機比較した結果、416→30.9FPS / 640→13.1FPSでFPSに2.4倍の差。検出精度は640が若干優れるが（低conf検出のconf改善、1件の追加検出）、FPSの低下が致命的なため416を採用。精度改善はフェーズ3の自前データ追加で対応する方針
- **imgsz=416版のエクスポート方法**: ラズパイ側で実行。過去に`benchmark_416.py`（現在は削除済み）内で`model_pt.export(format="ncnn", imgsz=416)`を実行してエクスポートした。再作成が必要な場合はラズパイで以下を実行:
  ```
  python -c "
  from ultralytics import YOLO
  model = YOLO('models/exp_001_best.pt')
  model.export(format='ncnn', imgsz=416)
  "
  ```
- **MJPEG配信採用理由**: SSH X11転送では5-10FPS程度に低下するため、JPEG圧縮+HTTP配信方式を採用。ブラウザで閲覧可能でデモにも適する
- **学習データ**: Roboflowのトランプカードデータセット（52クラス、10,100枚）。正面・均一照明が中心
- **train_exp_*.pyのimport方式**: sys.path.insertでスクリプト自身のディレクトリを追加。VSCode実行・コマンドライン実行の両方に対応

---

## 進捗・フェーズ管理

### 完了済み
- **フェーズ1**: ラズパイ5環境構築（picamera2でカメラ動作確認済み）
- **フェーズ2**: モデル学習 & ラズパイ最適化
  - 実験001: YOLO11n学習完了
  - 実験002: YOLO26n学習完了
  - FPS計測: YOLO11n NCNN形式で30.9FPS達成（目標30FPS超え）
  - YOLO26nはNCNN変換非対応と判明
  - imgsz 416 vs 640 比較実施 → 416採用を根拠付きで決定
  - GitHubにスクリプト・ドキュメント・README整備済み
  - ラズパイ側ファイル整理完了（scripts/フォルダに統合、不要ファイル削除）
- **MJPEG配信**: live_stream.py動作確認完了（約27FPS、imgsz=416で正常検出）

### 次にやること
- **フェーズ3**: 自前データ撮影（手持ち・斜めカード）→ 再学習で精度改善
  - **着手時に決めるべき未決定事項:**
    - 撮影枚数の目標（何枚くらい撮るか）
    - 撮影条件（照明・角度・背景のバリエーション）
    - 既存データセット（10,100枚）に追加するか、差し替えるか
    - train_config.pyのパラメータ変更の有無（shear/perspectiveの有効化等）
    - 学習imgsz=416に変更するか（現状640で学習→416でエクスポート）
    - アノテーションツールの選定（Roboflow / labelImg / CVAT等）
    - アノテーション形式: 既存データセットはYOLO形式（txt）。追加データも同じ形式で作成すること
    - ラズパイ側ultralyticsをグローバルpipからvenv内に移行するか
- **フェーズ4**: ブラックジャック戦略テーブル実装
  - ベーシックストラテジー（プレイヤー手札合計 × ディーラーアップカードの固定テーブル）をルックアップ実装
  - 検出結果からカード値を算出（A=1or11, J/Q/K=10, 数字=そのまま）
  - 画面にHIT/STAND/DOUBLE/SPLIT等の推奨アクションを表示
- **フェーズ5**: リアルタイムUI + デモ動画
  - ブラウザ上のUI拡張（現在のMJPEG配信ページを発展させる）
  - 戦略表示、検出カード一覧、手札合計値などをオーバーレイ表示
  - デモ動画を撮影してREADMEに掲載

### 将来の拡張アイデア（未検討）
- カードカウンティング機能（検出カード履歴の管理 → 残りデッキの偏りからベット調整）
- ポーカーのハンド判定など他のトランプゲームへの応用
- AWSなどクラウド上へのデプロイ（ポートフォリオとしての活用。実現可能性・有効性は未議論）

---

## 技術的注意事項

- **imgsz不整合に注意**: NCNNモデルはimgsz=416でエクスポート済み。推論時は必ずimgsz=416を指定すること。640を指定すると大量の誤検出が発生する
- **export_and_benchmark.pyに注意**: このスクリプトはIMGSZ_EXPORT=640でエクスポートする設定。再実行するとexp_001_best_ncnn_model/が640版で上書きされる。再エクスポートが必要な場合はIMGSZ_EXPORTを416に変更してから実行すること
- **NCNNフォルダ命名制約**: フォルダ名が`_ncnn_model`で終わる必要がある。これはultralyticsがモデル形式をフォルダ名から自動判別する仕様のため
- **MJPEG配信のアーキテクチャ**: Flaskは`host='0.0.0.0'`で起動（LAN内の全端末からアクセス可能）。検出スレッドが1つのグローバル変数にフレームを書き込み、Flaskスレッドが読み取って配信する構成。ブラウザ接続数が増えても検出FPSは低下しないが、配信帯域は分散される
- **picamera2のRGBA→BGR変換**: カメラ取得解像度は640x480（RGBA形式）。フレーム取得後に `cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)` が必要。忘れると色がおかしくなる。YOLO入力時にimgsz=416にリサイズされる（ultralyticsが自動処理）
- **PyTorch CPU版への入替済み**: ラズパイのPyTorchは2.11.0+cpu。04/11にCUDAビルド版（+cu130）からCPU版に入替。FPSに変化なし（NCNN推論はCPU完結のため）だがディスク容量削減・環境クリーン化のメリットあり
- **runs/の構造**: 以前は`runs/detect/runs/`という二重構造だった。train_exp_*.pyの`project`を`"runs/detect"`に修正し解消済み
- ラズパイ側で `cd ~/edge-ai-card-reader` してから実行する前提（MODEL_PATH等が相対パス。詳細はlive_stream.pyパラメータ一覧参照）
- NCNNモデル読み込み時に `task=detect` 未指定の警告が出るが動作に影響なし
- PowerShellでは日本語を含むgit commitメッセージが正しく閉じられない場合がある → 英語メッセージを推奨
- PyTorchはCUDA/CPU版の選択が環境依存のためrequirements.txtに含めず、コメントで別途インストール手順を記載

---

## 未解決・確認済みの疑問

本文の各セクションに反映済みの回答はここには残さない（肥大化防止）。
本文に反映しきれない補足情報のみ記録する。

| 質問 | 回答 | 日付 |
|------|------|------|
| ラズパイのPyTorchインストール経緯は？ | venv作成前にグローバルpipでCUDA版がインストール済みだった。04/11にグローバルから削除し、venv内にCPU版を再インストール | 2026/04/11 |
| PC側のPython環境構築で問題はあった？ | 最初`pip install ultralytics`がグローバルの3.13に入ってしまった。.venvを有効化し直してからインストール。またCPU版torchが入ったので`--index-url whl/cu118`でGPU版に入れ直した | 2026/04/11 |
| compare_imgsz.pyは今後使うか？ | imgsz比較は完了済み。フェーズ3で再学習後に再比較する可能性があるため保持。不要になればアーカイブ候補 | 2026/04/12 |
| exp_001_416_ncnn_modelとexp_001_best_ncnn_modelの違いは？ | 同一内容。416版のコピーとして比較用に作成。best_ncnn_modelがlive_stream.pyの参照先。416版は不要になれば削除可 | 2026/04/12 |

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026/04/10 | PROJECT_STRUCTURE.md 初版作成 |
| 2026/04/11 | live_stream.py追加・動作確認（約27FPS） |
| 2026/04/11 | imgsz 416vs640比較→416採用決定 |
| 2026/04/11 | ラズパイ側ファイル整理（scripts/統合）、全スクリプトのdocstring統一 |
| 2026/04/11 | train_exp_*.pyのsys.path修正、Gitブランチをmainに統一 |
| 2026/04/11 | requirements-pc.txt / requirements-raspi.txt作成 |
| 2026/04/11 | ラズパイvenv再構築（PyTorch CPU版に入替）、venv再構築手順追加 |
| 2026/04/11 | レビュー1回目: デモ実行手順集約、フェーズ4/5詳細化、実験採番ルール |
| 2026/04/11 | レビュー2回目: check_augmentation.py移動、スクリプト説明明確化 |
| 2026/04/11 | レビュー3回目: .gitignore/LICENSE記載、52クラス一覧追加、GPU/学習時間追記 |
| 2026/04/12 | runs二重構造解消、データセット分割比率追記、Q&Aセクション整理 |
| 2026/04/12 | data.yamlパス解決の仕組み明記、NCNNフォルダ命名制約の根拠追記 |
| 2026/04/12 | imgsz比較の判断根拠補足、デモURLにIPフォールバック復活、exp_002保持方針明記 |
